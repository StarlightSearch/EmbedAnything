from embed_anything import (
    EmbeddingModel,
    WhichModel,
    embed_query,
    embed_file,
    embed_directory,
)

import os


def test_jina_model_creation():

    model = EmbeddingModel.from_pretrained_hf(
        WhichModel.Jina,
        model_id="jinaai/jina-embeddings-v2-small-en",
        revision="main",
    )
    assert model is not None


def test_jina_model_query(jina_model):

    data = embed_query(["Photo of a monkey?"], jina_model)
    assert len(data) == 1
    assert data[0].embedding is not None
    assert len(data[0].embedding) == 512


def test_jina_model_file(jina_model):

    data = embed_file("test_files/test.pdf", jina_model)
    path = os.path.abspath("test_files/test.pdf")
    assert data[0].embedding is not None
    assert len(data[0].embedding) == 512


def test_jina_model_directory(jina_model):

    data = embed_directory("test_files", jina_model)
    assert data[0].embedding is not None
    assert len(data[0].embedding) == 512
