#!/usr/bin/env python3
"""
Script to verify and fix Gemma model download issues
"""

import os
import shutil
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def clear_cache():
    """Clear the Hugging Face cache for the specific model"""
    cache_path = Path.home() / ".cache" / "huggingface" / "hub" / "models--google--gemma-2-2b-it"
    if cache_path.exists():
        print(f"Clearing cache at: {cache_path}")
        shutil.rmtree(cache_path)
        print("Cache cleared successfully")
    else:
        print("No cache found to clear")

def download_model():
    """Download the model with proper error handling"""
    model_name = "google/gemma-2-2b-it"
    local_path = "./gemma-2-2b-it-local"
    
    try:
        print(f"Downloading {model_name}...")
        
        # Download tokenizer
        print("Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            trust_remote_code=True,
            cache_dir="./model_cache"
        )
        print("✓ Tokenizer downloaded successfully")
        
        # Download model
        print("Downloading model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map='auto',
            trust_remote_code=True,
            cache_dir="./model_cache"
        )
        print("✓ Model downloaded successfully")
        
        # Save locally to ensure complete download
        print(f"Saving model to {local_path}...")
        model.save_pretrained(local_path)
        tokenizer.save_pretrained(local_path)
        print(f"✓ Model saved locally to {local_path}")
        
        # Verify the saved files
        verify_local_model(local_path)
        
        return True
        
    except Exception as e:
        print(f"❌ Error downloading model: {e}")
        return False

def verify_local_model(local_path):
    """Verify that all necessary model files are present"""
    local_path = Path(local_path)
    
    if not local_path.exists():
        print(f"❌ Local path {local_path} does not exist")
        return False
    
    # Check for essential files
    essential_files = [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json"
    ]
    
    # Check for model weights (should have at least one safetensors file)
    safetensors_files = list(local_path.glob("*.safetensors"))
    
    print(f"Found {len(safetensors_files)} safetensors files:")
    for f in safetensors_files:
        print(f"  - {f.name}")
    
    if not safetensors_files:
        print("❌ No safetensors files found!")
        return False
    
    # Check essential files
    missing_files = []
    for file in essential_files:
        if not (local_path / file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing essential files: {missing_files}")
        return False
    
    print("✓ All essential files are present")
    return True

def main():
    """Main function to fix the model download"""
    print("=== Gemma Model Download Fix ===")
    
    # Step 1: Clear cache
    clear_cache()
    
    # Step 2: Download model
    if download_model():
        print("\n🎉 Model download completed successfully!")
        print("You can now run your training script.")
    else:
        print("\n❌ Model download failed. Please check the error messages above.")
        print("You may need to:")
        print("1. Check your internet connection")
        print("2. Ensure you have sufficient disk space")
        print("3. Try running the script again")

if __name__ == "__main__":
    main()
