"""
evaluate.py
Runs the fine-tuned model on the validation set and reports:
  - Exact-match accuracy (full JSON match)
  - Aspect F1   (precision/recall on predicted aspect sets)
  - Sentiment accuracy per aspect

Usage:
  # With LoRA adapter (not merged)
  python evaluate.py --model Qwen/Qwen3-4B --adapter outputs/qwen3-absa-qlora/final_adapter

  # With merged model
  python evaluate.py --model outputs/qwen3-absa-merged
"""

import argparse, json, re, torch
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm

VALID_ASPECTS    = {"food","service","ambiance","cleanliness","price","delivery","app_experience","general","none"}
VALID_SENTIMENTS = {"positive","negative","neutral"}

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",           default="Qwen/Qwen3-4B")
    p.add_argument("--adapter",         default=None,
                   help="Path to LoRA adapter directory (skip if model is already merged)")
    p.add_argument("--val_file",        default="data/val.jsonl")
    p.add_argument("--batch_size",      type=int, default=8)
    p.add_argument("--max_new_tokens",  type=int, default=256)
    p.add_argument("--use_4bit",        action="store_true", default=False)
    p.add_argument("--output_file",     default="eval_results.json")
    return p.parse_args()

def load_model(args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.adapter or args.model, trust_remote_code=True, padding_side="left"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    ) if args.use_4bit else None

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
        model = model.merge_and_unload()   # fuse for faster inference
    model.eval()
    return tokenizer, model

def extract_json(text: str):
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except Exception:
            pass
    return None

def generate_batch(tokenizer, model, prompts, max_new_tokens):
    inputs = tokenizer(prompts, return_tensors="pt", padding=True,
                       truncation=True, max_length=512).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=False, temperature=None, top_p=None,
            pad_token_id=tokenizer.eos_token_id,
        )
    decoded = []
    for i, out in enumerate(outputs):
        gen = out[inputs["input_ids"].shape[1]:]
        decoded.append(tokenizer.decode(gen, skip_special_tokens=True))
    return decoded

# ── Metrics ────────────────────────────────────────────────────────────────────
def aspect_f1(pred_aspects, gold_aspects):
    pred_set = set(pred_aspects)
    gold_set = set(gold_aspects)
    tp = len(pred_set & gold_set)
    prec = tp / len(pred_set) if pred_set else 0.0
    rec  = tp / len(gold_set) if gold_set else 0.0
    f1   = 2*prec*rec / (prec+rec) if (prec+rec) > 0 else 0.0
    return prec, rec, f1

def sentiment_accuracy(pred_sents, gold_sents):
    common = set(pred_sents) & set(gold_sents)
    if not common:
        return 0.0, 0
    correct = sum(1 for a in common if pred_sents[a] == gold_sents[a])
    return correct / len(common), len(common)

def main():
    args = parse_args()
    tokenizer, model = load_model(args)

    records = [json.loads(l) for l in open(args.val_file, encoding="utf-8")]
    print(f"Evaluating {len(records):,} validation samples...")

    all_prec, all_rec, all_f1 = [], [], []
    all_sent_acc, sent_total  = [], 0
    exact_matches, parse_fails = 0, 0
    per_sample = []

    for i in tqdm(range(0, len(records), args.batch_size)):
        batch = records[i : i + args.batch_size]

        # Build inference prompts (strip last assistant turn)
        prompts = []
        for s in batch:
            msgs = [m for m in s["messages"] if m["role"] != "assistant"]
            prompt = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
            prompts.append(prompt)

        outputs = generate_batch(tokenizer, model, prompts, args.max_new_tokens)

        for sample, raw_out in zip(batch, outputs):
            # Ground truth
            gt_text = sample["messages"][-1]["content"]
            gt = json.loads(gt_text)
            gold_aspects = gt["aspects"]
            gold_sents   = gt["aspect_sentiments"]

            # Prediction
            pred = extract_json(raw_out)
            if pred is None:
                parse_fails += 1
                pred_aspects, pred_sents = [], {}
            else:
                pred_aspects = pred.get("aspects", [])
                pred_sents   = pred.get("aspect_sentiments", {})

            # Exact match
            is_exact = (
                pred is not None and
                set(pred_aspects) == set(gold_aspects) and
                pred_sents == gold_sents
            )
            if is_exact:
                exact_matches += 1

            # Aspect F1
            prec, rec, f1 = aspect_f1(pred_aspects, gold_aspects)
            all_prec.append(prec); all_rec.append(rec); all_f1.append(f1)

            # Sentiment accuracy (on matched aspects only)
            sacc, sn = sentiment_accuracy(pred_sents, gold_sents)
            if sn > 0:
                all_sent_acc.append(sacc)
                sent_total += sn

            per_sample.append({
                "review_id":    sample["review_id"],
                "gold_aspects": gold_aspects,
                "pred_aspects": pred_aspects,
                "gold_sents":   gold_sents,
                "pred_sents":   pred_sents,
                "exact_match":  is_exact,
                "aspect_f1":    round(f1, 4),
                "sent_acc":     round(sacc, 4),
                "raw_output":   raw_out,
            })

    n = len(records)
    results = {
        "total_samples":    n,
        "parse_fail_count": parse_fails,
        "parse_fail_pct":   round(100 * parse_fails / n, 2),
        "exact_match_acc":  round(100 * exact_matches / n, 2),
        "aspect_precision": round(100 * sum(all_prec) / n, 2),
        "aspect_recall":    round(100 * sum(all_rec)  / n, 2),
        "aspect_f1":        round(100 * sum(all_f1)   / n, 2),
        "sentiment_acc":    round(100 * sum(all_sent_acc) / len(all_sent_acc), 2) if all_sent_acc else 0,
        "per_sample":       per_sample,
    }

    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n══════════════════════════════════")
    print(f"  Exact Match Accuracy : {results['exact_match_acc']}%")
    print(f"  Aspect Precision     : {results['aspect_precision']}%")
    print(f"  Aspect Recall        : {results['aspect_recall']}%")
    print(f"  Aspect F1            : {results['aspect_f1']}%")
    print(f"  Sentiment Accuracy   : {results['sentiment_acc']}%")
    print(f"  Parse Failures       : {results['parse_fail_count']} ({results['parse_fail_pct']}%)")
    print(f"══════════════════════════════════")
    print(f"Full results → {args.output_file}")

if __name__ == "__main__":
    main()
