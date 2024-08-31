import re
from typing import Dict, List
import uuid
import time
import embed_anything
from embed_anything import EmbeddingModel, WhichModel
from embed_anything.vectordb import Adapter
from pinecone import Pinecone, ServerlessSpec
import os


model = EmbeddingModel.from_pretrained_local(
    WhichModel.Bert, model_id="prithivida/miniMiracle_te_v1"
)

data = embed_anything.embed_file("test_files/test.pdf", embeder=model)
print(data[0].embedding)
