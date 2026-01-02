#!/usr/bin/env python3
"""
Script to run embed_store with viber1/indian-law-dataset
"""
from datasets import load_dataset
from search_r1.search.lance_retrieval import embed_store

# Load the dataset from HuggingFace
print("Loading dataset viber1/indian-law-dataset...")
ds = load_dataset("viber1/indian-law-dataset")

print(f"Dataset loaded. Splits: {ds.keys()}")
# Limit to first 100 rows to reduce embedding time/size
ds = {"train": ds["train"].select(range(100))}
print(f"Train split size: {len(ds['train'])}")

# Check if the dataset has the expected structure
if len(ds['train']) > 0:
    sample = ds['train'][0]
    print(f"Sample keys: {sample.keys()}")
    if 'Response' not in sample:
        print("Warning: 'Response' field not found in dataset. Available fields:", list(sample.keys()))
        # Try to find a text field
        text_fields = [k for k in sample.keys() if 'text' in k.lower() or 'response' in k.lower() or 'content' in k.lower()]
        if text_fields:
            print(f"Found potential text fields: {text_fields}")

# Run embed_store
print("\nRunning embed_store...")
table = embed_store(ds, lancedb_path="tmp/lancedb", table_name="docs")

print("\nDone! Embeddings stored successfully.")

