#!/bin/bash

# Fix for Gemma model download issues
echo "Clearing Hugging Face cache..."
rm -rf ~/.cache/huggingface/hub/models--google--gemma-2-2b-it

echo "Re-downloading the model..."
python3 -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Force re-download
model_name = 'google/gemma-2-2b-it'
print(f'Downloading {model_name}...')

# Download tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
print('Tokenizer downloaded successfully')

# Download model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map='auto',
    trust_remote_code=True
)
print('Model downloaded successfully')

# Save locally to ensure complete download
local_path = './gemma-2-2b-it-local'
model.save_pretrained(local_path)
tokenizer.save_pretrained(local_path)
print(f'Model saved locally to {local_path}')
"

echo "Model download completed!"
