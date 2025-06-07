from embed_anything import (
    EmbeddingModel,
    TextEmbedConfig,
    WhichModel,
    embed_query,
    embed_file,
    embed_directory,
    ONNXModel,
)

import os
import pytest
import tempfile
import itertools

# Global test parameters
MODEL_FIXTURES = ["bert_model", "onnx_model"]
CONFIGS = [None, TextEmbedConfig(batch_size=32, chunk_size=1000)]
ALL_COMBINATIONS = list(itertools.product(MODEL_FIXTURES, CONFIGS))

# Define common parametrize decorator
model_fixture_parametrize = pytest.mark.parametrize("model_fixture", MODEL_FIXTURES)
model_and_config_parametrize = pytest.mark.parametrize(
    "model_fixture,config", ALL_COMBINATIONS
)


@model_and_config_parametrize
def test_bert_model_file(model_fixture, config, test_pdf_file, request):
    model = request.getfixturevalue(model_fixture)
    data = model.embed_file(test_pdf_file, config)
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


def test_onnx_model_creation():
    model = EmbeddingModel.from_pretrained_onnx(
        WhichModel.Bert, ONNXModel.AllMiniLML6V2Q
    )
    assert model is not None


@model_fixture_parametrize
def test_bert_model_query(model_fixture, request):
    model = request.getfixturevalue(model_fixture)
    data = embed_query(["Photo of a monkey?"], model)
    assert len(data) == 1
    assert data[0].embedding is not None
    assert len(data[0].embedding) == 384


@model_and_config_parametrize
def test_bert_model_directory(model_fixture, config, test_text_directory, request):
    model = request.getfixturevalue(model_fixture)
    data = embed_directory(test_text_directory, model, config=config)
    assert data[0].embedding is not None
    assert len(data[0].embedding) == 384


@model_fixture_parametrize
def test_bert_model_empty_query(model_fixture, request):
    model = request.getfixturevalue(model_fixture)
    data = embed_query([""], model)
    assert len(data) == 1
    assert data[0].embedding is not None
    assert len(data[0].embedding) == 384


@model_fixture_parametrize
def test_bert_model_long_query(model_fixture, request):
    model = request.getfixturevalue(model_fixture)
    long_text = " ".join(["long"] * 1000)
    data = embed_query([long_text], model)
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
