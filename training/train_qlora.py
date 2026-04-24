"""
train_qlora.py
QLoRA fine-tuning of Qwen3-4B (non-thinking) for Aspect-Based Sentiment Analysis.

Quick start (labeled data only):
  python train_qlora.py

With pseudo-labeled data (run pseudo_label.py first):
  python train_qlora.py --train_file data/train_combined.jsonl

Resume from checkpoint:
  python train_qlora.py --resume_from_checkpoint outputs/checkpoint-500
"""

import argparse, json, os
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

# ── CLI ────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",           default="Qwen/Qwen3-4B")
    p.add_argument("--train_file",      default="data/train.jsonl",
                   help="Use data/train_combined.jsonl for pseudo-labeled version")
    p.add_argument("--val_file",        default="data/val.jsonl")
    p.add_argument("--output_dir",      default="outputs/qwen3-absa-qlora")
    p.add_argument("--max_seq_len",     type=int,   default=512)
    p.add_argument("--per_device_bs",   type=int,   default=4)
    p.add_argument("--grad_accum",      type=int,   default=4,
                   help="Effective batch = per_device_bs * grad_accum * n_gpus")
    p.add_argument("--epochs",          type=int,   default=5)
    p.add_argument("--lr",              type=float, default=2e-4)
    p.add_argument("--lora_r",          type=int,   default=16)
    p.add_argument("--lora_alpha",      type=int,   default=32)
    p.add_argument("--lora_dropout",    type=float, default=0.05)
    p.add_argument("--use_4bit",        action="store_true", default=True,
                   help="QLoRA (4-bit). Pass --no-use_4bit for full LoRA in 8-bit")
    p.add_argument("--no-use_4bit",     dest="use_4bit", action="store_false")
    p.add_argument("--resume_from_checkpoint", default=None)
    return p.parse_args()

# ── Data helpers ───────────────────────────────────────────────────────────────
def load_jsonl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]

def apply_template(sample, tokenizer):
    """
    Convert messages list → single formatted string.
    enable_thinking=False disables Qwen3 chain-of-thought.
    """
    return tokenizer.apply_chat_template(
        sample["messages"],
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,        # Non-thinking variant
    )

def build_dataset(records, tokenizer):
    texts = [apply_template(r, tokenizer) for r in records]
    return Dataset.from_dict({"text": texts})

# ── Model / tokenizer ──────────────────────────────────────────────────────────
def load_model_and_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
        padding_side="right",          # Required for SFT with left-pad models
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = None
    if args.use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if not args.use_4bit else None,
        attn_implementation="flash_attention_2",   # remove if not installed
    )

    if args.use_4bit:
        model = prepare_model_for_kbit_training(model)

    return model, tokenizer

# ── LoRA config ────────────────────────────────────────────────────────────────
def get_lora_config(args):
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        # Target all attention + MLP projections for best ABSA performance
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        inference_mode=False,
    )

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading model: {args.model}")
    model, tokenizer = load_model_and_tokenizer(args)
    model = get_peft_model(model, get_lora_config(args))
    model.print_trainable_parameters()

    print("Building datasets...")
    train_records = load_jsonl(args.train_file)
    val_records   = load_jsonl(args.val_file)
    train_ds = build_dataset(train_records, tokenizer)
    val_ds   = build_dataset(val_records,   tokenizer)
    print(f"  Train: {len(train_ds):,}  |  Val: {len(val_ds):,}")

    # Only compute loss on the assistant turn (not prompt)
    # The response template is the assistant header token sequence in Qwen3 chat format
    response_template = "<|im_start|>assistant\n"
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer,
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_bs,
        per_device_eval_batch_size=args.per_device_bs * 2,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,

        # Mixed precision
        bf16=True,
        fp16=False,

        # Logging & evaluation
        logging_steps=20,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        # Misc
        dataloader_num_workers=4,
        group_by_length=True,          # Speeds up training
        report_to="none",              # Change to "wandb" if you want tracking
        run_name="qwen3-absa-qlora",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        max_seq_length=args.max_seq_len,
        dataset_text_field="text",
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )

    print("Starting training...")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # Save final adapter
    final_path = os.path.join(args.output_dir, "final_adapter")
    trainer.model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"\nFinal LoRA adapter saved → {final_path}")
    print("To merge weights run:  python merge_adapter.py")

if __name__ == "__main__":
    main()
