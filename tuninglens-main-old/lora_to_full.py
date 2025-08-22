from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftConfig, PeftModel

import argparse

parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model and save full model.")
parser.add_argument("dir", type=str, help="Base directory containing all versions")
parser.add_argument("--last", action="store_true", help="Select the final checkpoint instead of the best (second to last) checkpoint")
args = parser.parse_args()

base_dir = Path(args.dir)

# Find all version directories (e.g., v1-20250715-030156)
version_dirs = sorted([d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith("v")])
if not version_dirs:
    raise FileNotFoundError(f"No version directories found in {base_dir}")

# Use the latest version (sorted lexicographically, which works for this naming scheme)
latest_version_dir = version_dirs[-1]

# Find all checkpoint directories in the latest version
ckpt_dirs = sorted([d for d in latest_version_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")])



# Extract checkpoint numbers and sort
ckpt_nums = sorted([int(d.name.split("-")[-1]) for d in ckpt_dirs])

if args.last:
    # Use the last (largest) checkpoint
    selected_ckpt_num = ckpt_nums[-1]
else:
    # Use the "best" checkpoint (second largest)
    if len(ckpt_dirs) < 2:
        raise FileNotFoundError(f"Not enough checkpoints found in {latest_version_dir}")
    selected_ckpt_num = ckpt_nums[-2]

selected_ckpt_dir = latest_version_dir / f"checkpoint-{selected_ckpt_num}"

adapter_path = str(selected_ckpt_dir)

# Automatically generate out_dir based on adapter_path
adapter_path_obj = Path(adapter_path.rstrip("/"))
# Go up two levels (e.g., .../mbpp-sanitized-train/v1-20250711-060256/checkpoint-30/ -> .../mbpp-sanitized-train/v1-20250711-060256/)
parent_dir = adapter_path_obj.parent.parent
# Name for merged model directory
out_dir = str(parent_dir) + "-full/"

# 1. Get adapter config to learn which base model it was trained on
cfg = PeftConfig.from_pretrained(adapter_path)

# 2. Load the base model in the same dtype it was trained with
base = AutoModelForCausalLM.from_pretrained(
    cfg.base_model_name_or_path,
    torch_dtype=torch.bfloat16       # or float32 if that’s how you trained
)

# 3. Attach adapter and merge → plain HF model (no PEFT wrappers)
model = PeftModel.from_pretrained(base, adapter_path)
model = model.merge_and_unload()    # becomes a regular `AutoModelForCausalLM`

# 4. Save merged model + tokenizer
Path(out_dir).mkdir(parents=True, exist_ok=True)
model.save_pretrained(out_dir, safe_serialization=True)   # saves as .safetensors
tok = AutoTokenizer.from_pretrained(cfg.base_model_name_or_path, use_fast=True)
tok.save_pretrained(out_dir)

print(f"Full model written to: {out_dir}")
