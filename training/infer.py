"""
infer.py
Quick inference demo — pass a review and get structured JSON back.

  python infer.py --model outputs/qwen3-absa-merged
  python infer.py --model Qwen/Qwen3-4B --adapter outputs/qwen3-absa-qlora/final_adapter
"""

import argparse, json, re, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

SYSTEM_PROMPT = (
    "You are an expert Aspect-Based Sentiment Analysis (ABSA) system. "
    "Given a customer review, identify ALL mentioned aspects and classify each sentiment as "
    "positive, negative, or neutral. "
    "Valid aspects are: food, service, ambiance, cleanliness, price, delivery, app_experience, general, none. "
    "Respond ONLY with a valid JSON object in exactly this format, no extra text:\n"
    '{"aspects": ["aspect1", "aspect2"], "aspect_sentiments": {"aspect1": "positive", "aspect2": "negative"}}'
)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",   default="outputs/qwen3-absa-merged")
    p.add_argument("--adapter", default=None)
    p.add_argument("--reviews", nargs="+",
                   default=["المكان نضيف وجميل والخدمة ممتازة بس الأكل كان وسط"])
    return p.parse_args()

def load_model(args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.adapter or args.model, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
        model = model.merge_and_unload()
    model.eval()
    return tokenizer, model

def analyze(review_text, tokenizer, model):
    messages = [
        {"role": "system",  "content": SYSTEM_PROMPT},
        {"role": "user",    "content": f"Analyze this review:\n{review_text}"},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=256,
            do_sample=False, temperature=None, top_p=None,
            pad_token_id=tokenizer.eos_token_id,
        )
    gen = out[0][inputs["input_ids"].shape[1]:]
    raw = tokenizer.decode(gen, skip_special_tokens=True).strip()

    # Parse
    try:
        return json.loads(raw)
    except Exception:
        m = re.search(r'\{.*\}', raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except Exception:
                pass
    return {"raw_output": raw, "parse_error": True}

def main():
    args = parse_args()
    tokenizer, model = load_model(args)
    for review in args.reviews:
        print(f"\nReview : {review}")
        result = analyze(review, tokenizer, model)
        print(f"Output : {json.dumps(result, ensure_ascii=False, indent=2)}")

if __name__ == "__main__":
    main()
