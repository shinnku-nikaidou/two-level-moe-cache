#!/usr/bin/env python3

# Script to download GPT-OSS models from Hugging Face
# usage: python scripts/download_llm.py
import os

from huggingface_hub import snapshot_download

REPO_ID = "openai/gpt-oss-20b"
DEST = os.path.join(os.path.dirname(__file__), "..", "data", "models", "gpt-oss-20b")

os.makedirs(DEST, exist_ok=True)
local_path = snapshot_download(
    repo_id=REPO_ID,
    local_dir=DEST,
    resume_download=True,
)
print(f"Downloaded to: {local_path}")


REPO_ID = "openai/gpt-oss-120b"
DEST = os.path.join(os.path.dirname(__file__), "..", "data", "models", "gpt-oss-120b")

os.makedirs(DEST, exist_ok=True)
local_path = snapshot_download(
    repo_id=REPO_ID,
    local_dir=DEST,
    resume_download=True,
)
print(f"Downloaded to: {local_path}")


REPO_ID = "microsoft/Phi-tiny-MoE-instruct"
DEST = os.path.join(os.path.dirname(__file__), "..", "data", "models", "phi-tiny-moe")

os.makedirs(DEST, exist_ok=True)
local_path = snapshot_download(
    repo_id=REPO_ID,
    local_dir=DEST,
    resume_download=True,
)

print(f"Downloaded to: {local_path}")
