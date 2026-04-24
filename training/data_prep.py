"""
data_prep.py
Converts the DeepX xlsx files into JSONL chat format for QLoRA fine-tuning.
Outputs:
  - data/train.jsonl        (labeled train data)
  - data/pseudo_ready.jsonl (unlabeled reviews formatted for pseudo-labeling)
  - data/val.jsonl          (validation data)
"""

import os, json, ast
import pandas as pd

os.makedirs("data", exist_ok=True)

SYSTEM_PROMPT = (
    "You are an expert Aspect-Based Sentiment Analysis (ABSA) system. "
    "Given a customer review, identify ALL mentioned aspects and classify each sentiment as "
    "positive, negative, or neutral. "
    "Valid aspects are: food, service, ambiance, cleanliness, price, delivery, app_experience, general, none. "
    "Respond ONLY with a valid JSON object in exactly this format, no extra text:\n"
    '{"aspects": ["aspect1", "aspect2"], "aspect_sentiments": {"aspect1": "positive", "aspect2": "negative"}}'
)

def make_user_msg(review_text: str) -> str:
    return f"Analyze this review:\n{review_text.strip()}"

def parse_field(val):
    if isinstance(val, (list, dict)):
        return val
    if isinstance(val, str):
        return ast.literal_eval(val)
    return val

def build_labeled_sample(row) -> dict:
    aspects = parse_field(row["aspects"])
    sentiments = parse_field(row["aspect_sentiments"])
    assistant_reply = json.dumps(
        {"aspects": aspects, "aspect_sentiments": sentiments},
        ensure_ascii=False
    )
    return {
        "review_id": int(row["review_id"]),
        "messages": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": make_user_msg(row["review_text"])},
            {"role": "assistant", "content": assistant_reply},
        ]
    }

def build_inference_sample(row) -> dict:
    return {
        "review_id": int(row["review_id"]),
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": make_user_msg(row["review_text"])},
        ]
    }

def write_jsonl(records, path):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  Wrote {len(records):,} records → {path}")

# ── Labeled train ──────────────────────────────────────────────────────────────
print("Processing train data...")
train_df = pd.read_excel("DeepX_train_cleaned.xlsx")
train_records = [build_labeled_sample(row) for _, row in train_df.iterrows()]
write_jsonl(train_records, "data/train.jsonl")

# ── Unlabeled (for pseudo-labeling) ───────────────────────────────────────────
print("Processing unlabeled data...")
unlabeled_df = pd.read_excel("DeepX_unlabeled_cleaned.xlsx")
unlabeled_records = [build_inference_sample(row) for _, row in unlabeled_df.iterrows()]
write_jsonl(unlabeled_records, "data/pseudo_ready.jsonl")

# ── Validation ────────────────────────────────────────────────────────────────
print("Processing validation data...")
val_df = pd.read_excel("DeepX_validation.xlsx")
val_records = [build_labeled_sample(row) for _, row in val_df.iterrows()]
write_jsonl(val_records, "data/val.jsonl")

print("\nDone. Files in ./data/")
