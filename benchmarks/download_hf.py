"""Download only the files approved: SST-2 validation, DistilBERT SST-2 (safetensors),
SegFormer-b0 ADE (safetensors), ADE20K validation parquet. Into the Hugging Face cache.

These are downloaded from the Hugging Face Hub, not redistributed by metric-diff; their own
licences and terms of use apply (ADE20K has terms of use for research purposes)."""
import os
import warnings
warnings.filterwarnings("ignore")

from huggingface_hub import hf_hub_download

files = [
    ("dataset", "stanfordnlp/sst2", "data/validation-00000-of-00001.parquet"),
    ("model", "distilbert/distilbert-base-uncased-finetuned-sst-2-english", "config.json"),
    ("model", "distilbert/distilbert-base-uncased-finetuned-sst-2-english", "tokenizer_config.json"),
    ("model", "distilbert/distilbert-base-uncased-finetuned-sst-2-english", "vocab.txt"),
    ("model", "distilbert/distilbert-base-uncased-finetuned-sst-2-english", "model.safetensors"),
    ("model", "nvidia/segformer-b0-finetuned-ade-512-512", "config.json"),
    ("model", "nvidia/segformer-b0-finetuned-ade-512-512", "preprocessor_config.json"),
    ("model", "nvidia/segformer-b0-finetuned-ade-512-512", "model.safetensors"),
    ("dataset", "1aurent/ADE20K", "data/validation-00000-of-00001.parquet"),
]
total = 0
for kind, repo, name in files:
    path = hf_hub_download(repo, name, repo_type=kind)
    size = os.path.getsize(path)
    total += size
    print(f"{size / 2**20:9.2f} MB  {repo}/{name}")
print(f"{total / 2**20:9.2f} MB  total")
