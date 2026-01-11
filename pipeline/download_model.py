"""Download a Hugging Face model repo snapshot into pipeline/models.

Usage: set HF_HUB_TOKEN env var if needed, then run this script.
"""
from huggingface_hub import snapshot_download
from pathlib import Path
import os

REPO_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
OUT = Path(__file__).resolve().parents[1] / 'models' / REPO_ID.split('/')[-1]
OUT.parent.mkdir(parents=True, exist_ok=True)

print(f"Starting snapshot download of {REPO_ID} into {OUT}")
path = snapshot_download(repo_id=REPO_ID, local_dir=str(OUT), allow_patterns=None)
print('Downloaded to', path)
