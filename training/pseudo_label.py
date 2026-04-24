"""
pseudo_label.py
Uses the BASE Qwen3-4B model (or your checkpoint) to generate pseudo-labels
for the unlabeled reviews, then merges them with the labeled train set.

Run AFTER data_prep.py and BEFORE (or between) training rounds:
  python pseudo_label.py --model Qwen/Qwen3-4B --batch_size 8

Outputs:
  data/pseudo_labeled.jsonl   ← labeled by the model
  data/train_combined.jsonl   ← original train + pseudo-labels (deduped)
"""

import argparse, json, re, torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

VALID_ASPECTS   = {"food","service","ambiance","cleanliness","price","delivery","app_experience","general","none"}
VALID_SENTIMENTS = {"positive","negative","neutral"}

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",       default="Qwen/Qwen3-4B",
                   help="HF model id or local checkpoint path")
    p.add_argument("--input",       default="data/pseudo_ready.jsonl")
    p.add_argument("--train",       default="data/train.jsonl")
    p.add_argument("--out_pseudo",  default="data/pseudo_labeled.jsonl")
    p.add_argument("--out_combined",default="data/train_combined.jsonl")
    p.add_argument("--batch_size",  type=int, default=8)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--confidence_filter", action="store_true",
                   help="Skip samples where model output is not clean JSON")
    return p.parse_args()

def load_model(model_id):
    print(f"Loading model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return tokenizer, model

def extract_json(text: str):
    """Extract and validate JSON from model output."""
    # Try direct parse first
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    # Try to find JSON block
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except Exception:
            pass
    return None

def validate_output(parsed) -> bool:
    if not isinstance(parsed, dict):
        return False
    if "aspects" not in parsed or "aspect_sentiments" not in parsed:
        return False
    aspects = parsed["aspects"]
    sents   = parsed["aspect_sentiments"]
    if not isinstance(aspects, list) or not isinstance(sents, dict):
        return False
    if not all(a in VALID_ASPECTS for a in aspects):
        return False
    if not all(s in VALID_SENTIMENTS for s in sents.values()):
        return False
    if set(aspects) != set(sents.keys()):
        return False
    return True

def generate_batch(tokenizer, model, samples, max_new_tokens):
    texts = [
        tokenizer.apply_chat_template(
            s["messages"],
            tokenize=False,
            add_generation_prompt=True,
            # Disable thinking for Qwen3
            enable_thinking=False,
        )
        for s in samples
    ]
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True,
                       max_length=512).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id,
        )
    # Decode only generated tokens
    results = []
    for i, out in enumerate(outputs):
        gen = out[inputs["input_ids"].shape[1]:]
        results.append(tokenizer.decode(gen, skip_special_tokens=True))
    return results

def main():
    args = parse_args()

    tokenizer, model = load_model(args.model)

    samples = [json.loads(l) for l in open(args.input, encoding="utf-8")]
    print(f"Pseudo-labeling {len(samples):,} samples...")

    pseudo_records, skipped = [], 0
    for i in tqdm(range(0, len(samples), args.batch_size)):
        batch = samples[i : i + args.batch_size]
        outputs = generate_batch(tokenizer, model, batch, args.max_new_tokens)

        for sample, out_text in zip(batch, outputs):
            parsed = extract_json(out_text)
            if parsed is None or not validate_output(parsed):
                skipped += 1
                if args.confidence_filter:
                    continue
                # Fallback: assign "none"/"neutral"
                parsed = {"aspects": ["none"], "aspect_sentiments": {"none": "neutral"}}

            # Build a labeled record matching train.jsonl format
            assistant_reply = json.dumps(parsed, ensure_ascii=False)
            record = {
                "review_id": sample["review_id"],
                "messages": sample["messages"] + [
                    {"role": "assistant", "content": assistant_reply}
                ]
            }
            pseudo_records.append(record)

    print(f"Generated {len(pseudo_records):,} pseudo-labels | skipped {skipped} (bad JSON)")

    with open(args.out_pseudo, "w", encoding="utf-8") as f:
        for r in pseudo_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Saved → {args.out_pseudo}")

    # Merge with original labeled train
    train_records = [json.loads(l) for l in open(args.train, encoding="utf-8")]
    seen_ids = {r["review_id"] for r in train_records}
    added = 0
    combined = train_records[:]
    for r in pseudo_records:
        if r["review_id"] not in seen_ids:
            combined.append(r)
            seen_ids.add(r["review_id"])
            added += 1

    with open(args.out_combined, "w", encoding="utf-8") as f:
        for r in combined:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Combined train: {len(combined):,} records (+{added} pseudo) → {args.out_combined}")

if __name__ == "__main__":
    main()
