from embed_anything import (
    EmbeddingModel,
    TextEmbedConfig,
    WhichModel,
    embed_query,
    embed_file,
    embed_directory,
)

import os
import pytest
import tempfile


@pytest.mark.parametrize(
    "config", [None, TextEmbedConfig(batch_size=32, chunk_size=256)]
)
def test_bert_model_file(bert_model, config, test_pdf_file):
    data = embed_file(test_pdf_file, bert_model, config)
    path = os.path.abspath(test_pdf_file)

    assert len(data) > 0
    assert data[0].embedding is not None
    assert len(data[0].embedding) == 384
    assert data[0].metadata["file_name"] == path


def test_bert_model_creation():

    model = EmbeddingModel.from_pretrained_hf(
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


@pytest.mark.parametrize(
    "config", [None, TextEmbedConfig(batch_size=32, chunk_size=256)]
)
def test_bert_model_directory(bert_model, config, test_text_directory):

    data = embed_directory(test_text_directory, bert_model, config=config)
    assert len(data) == 243
    assert data[0].embedding is not None
    assert len(data[0].embedding) == 384


def test_bert_model_empty_query(bert_model):
    data = embed_query([""], bert_model)
    assert len(data) == 1
    assert data[0].embedding is not None
    assert len(data[0].embedding) == 384


def test_bert_model_long_query(bert_model):
    long_text = " ".join(["long"] * 1000)
    data = embed_query([long_text], bert_model)
    assert len(data) == 1
    assert data[0].embedding is not None
    assert len(data[0].embedding) == 384


def test_bert_model_non_ascii_query(bert_model):
    non_ascii_text = "こんにちは世界"
    data = embed_query([non_ascii_text], bert_model)
    assert len(data) == 1
    assert data[0].embedding is not None
    assert len(data[0].embedding) == 384


def test_bert_model_nonexistent_file(bert_model):
    with pytest.raises(FileNotFoundError):
        embed_file("nonexistent_file.txt", bert_model)


def test_bert_model_empty_directory(bert_model, tmp_path):
    empty_dir = tmp_path / "empty_dir"
    empty_dir.mkdir()
    data = embed_directory(str(empty_dir), bert_model)
    assert len(data) == 0


def test_bert_model_unsupported_file_type(bert_model, tmp_path):

    # Create a file with an unsupported extension
    with open(tmp_path / "unsupported.mp3", "w") as f:
        f.write("This is a test file")

    with pytest.raises(ValueError):
        embed_file(str(tmp_path / "unsupported.mp3"), bert_model)