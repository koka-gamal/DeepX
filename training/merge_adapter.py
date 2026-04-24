"""
merge_adapter.py
Merges the LoRA adapter weights into the base model for faster inference.
Run this once after training is done.

  python merge_adapter.py \
    --base   Qwen/Qwen3-4B \
    --adapter outputs/qwen3-absa-qlora/final_adapter \
    --output  outputs/qwen3-absa-merged
"""

import argparse, json, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base",    default="Qwen/Qwen3-4B")
    p.add_argument("--adapter", default="outputs/qwen3-absa-qlora/final_adapter")
    p.add_argument("--output",  default="outputs/qwen3-absa-merged")
    return p.parse_args()

def main():
    args = parse_args()
    print(f"Loading base model: {args.base}")
    tokenizer = AutoTokenizer.from_pretrained(args.adapter, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True
    )
    print(f"Loading adapter: {args.adapter}")
    model = PeftModel.from_pretrained(model, args.adapter)
    print("Merging weights...")
    model = model.merge_and_unload()
    print(f"Saving merged model → {args.output}")
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    print("Done.")

if __name__ == "__main__":
    main()
