# DeepX

Hackathon repository for multilingual review cleaning and ABSA training.

## Repository Structure

- `data/cleaned/`: cleaned datasets used for training and inference input
- `data/raw/`: raw files intentionally kept unchanged (currently validation)
- `training/`: model training and evaluation scripts
- `training/data/`: generated JSONL files from `training/data_prep.py`

## Data Notes

- Reviews are kept untranslated.
- Emojis and multilingual text are preserved.
- Validation is intentionally not cleaned (used as provided by challenge guidance).

## Aspect Taxonomy

Use exactly these aspect labels:

- `food`
- `service`
- `price`
- `cleanliness`
- `delivery`
- `ambiance`
- `app_experience`
- `general`
- `none`

## Quick Start

From repo root:

```bash
cd training
pip install -r requirements.txt
python data_prep.py
```

This generates:

- `training/data/train.jsonl`
- `training/data/pseudo_ready.jsonl`
- `training/data/val.jsonl`
