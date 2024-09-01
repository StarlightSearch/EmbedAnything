import re
from typing import Dict, List
import uuid
import time
import embed_anything
from embed_anything import EmbeddingModel, TextEmbedConfig, WhichModel
from embed_anything.vectordb import Adapter
import os


model = EmbeddingModel.from_pretrained_hf(
    WhichModel.Bert, model_id="sentence-transformers/all-MiniLM-L6-v2"
)
config = TextEmbedConfig(chunk_size=200, batch_size=32)
data = embed_anything.embed_file("test_files/test.pdf", embeder=model, config=config)
print(data[0].embedding)
