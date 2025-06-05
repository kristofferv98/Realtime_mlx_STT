#!/usr/bin/env python3
"""Test model downloading for MLX Whisper."""

from huggingface_hub import snapshot_download
import os

repo_id = "openai/whisper-large-v3-turbo"
print(f"Testing download of {repo_id}...")

try:
    model_path = snapshot_download(
        repo_id=repo_id,
        allow_patterns=["config.json", "model.safetensors", "tokenizer.json", "*.json", "*.txt"],
        local_files_only=False
    )
    print(f"✅ Model downloaded successfully to: {model_path}")
    
    # Check if files exist
    config_path = os.path.join(model_path, 'config.json')
    model_file = os.path.join(model_path, 'model.safetensors')
    
    if os.path.exists(config_path):
        print("✅ config.json exists")
    else:
        print("❌ config.json missing")
        
    if os.path.exists(model_file):
        print("✅ model.safetensors exists")
    else:
        print("❌ model.safetensors missing")
        
    # List all files
    print("\nAll files in model directory:")
    for file in os.listdir(model_path):
        print(f"  - {file}")
        
except Exception as e:
    print(f"❌ Error downloading model: {e}")