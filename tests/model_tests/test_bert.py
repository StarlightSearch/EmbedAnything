from embed_anything import (
    EmbeddingModel,
    WhichModel,
    embed_query,
    embed_file,
    embed_directory,
)

import os


def test_bert_model_creation():
    model = EmbeddingModel.from_pretrained_local(
        WhichModel.Bert,
        model_id="sentence-transformers/all-MiniLM-L6-v2",
        revision="main",
    )

    assert model is not None


def test_bert_model_query(bert_model):

    data = embed_query(["Photo of a monkey?"], bert_model)
    assert len(data) == 1
    assert data[0].embedding is not None
    assert len(data[0].embedding) == 384


def test_bert_model_file(bert_model):
    data = embed_file("test_files/test.pdf", bert_model)

    path = os.path.abspath("test_files/test.pdf")
    assert data[0].embedding is not None
    assert len(data[0].embedding) == 384
    assert data[0].metadata["file_name"] == path
