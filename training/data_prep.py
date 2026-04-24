"""
data_prep.py
Converts the DeepX xlsx files into JSONL chat format for QLoRA fine-tuning.

Inputs:
  - ../data/cleaned/DeepX_train_cleaned.xlsx
  - ../data/cleaned/DeepX_unlabeled_cleaned.xlsx
  - ../data/raw/DeepX_validation.xlsx

Outputs:
  - ./data/train.jsonl
  - ./data/pseudo_ready.jsonl
  - ./data/val.jsonl
"""

import ast
import json
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
OUTPUT_DIR = SCRIPT_DIR / "data"
CLEANED_DIR = REPO_ROOT / "data" / "cleaned"
RAW_DIR = REPO_ROOT / "data" / "raw"

OUTPUT_DIR.mkdir(exist_ok=True)

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
        ensure_ascii=False,
    )
    return {
        "review_id": int(row["review_id"]),
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": make_user_msg(row["review_text"])},
            {"role": "assistant", "content": assistant_reply},
        ],
    }


def build_inference_sample(row) -> dict:
    return {
        "review_id": int(row["review_id"]),
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": make_user_msg(row["review_text"])},
        ],
    }


def write_jsonl(records, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"  Wrote {len(records):,} records -> {path}")


print("Processing train data...")
train_df = pd.read_excel(CLEANED_DIR / "DeepX_train_cleaned.xlsx")
train_records = [build_labeled_sample(row) for _, row in train_df.iterrows()]
write_jsonl(train_records, OUTPUT_DIR / "train.jsonl")

print("Processing unlabeled data...")
unlabeled_df = pd.read_excel(CLEANED_DIR / "DeepX_unlabeled_cleaned.xlsx")
unlabeled_records = [build_inference_sample(row) for _, row in unlabeled_df.iterrows()]
write_jsonl(unlabeled_records, OUTPUT_DIR / "pseudo_ready.jsonl")

print("Processing validation data...")
val_df = pd.read_excel(RAW_DIR / "DeepX_validation.xlsx")
val_records = [build_labeled_sample(row) for _, row in val_df.iterrows()]
write_jsonl(val_records, OUTPUT_DIR / "val.jsonl")

print(f"\nDone. Files in {OUTPUT_DIR}")
