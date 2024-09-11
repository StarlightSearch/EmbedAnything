from embed_anything import (
    EmbeddingModel,
    TextEmbedConfig,
    WhichModel,
    embed_query,
    embed_file,
    embed_directory,
    embed_image_directory,
)
import pytest
import os


def test_clip_model_creation():
    model = EmbeddingModel.from_pretrained_hf(
        WhichModel.Clip, model_id="openai/clip-vit-base-patch32", revision="refs/pr/15"
    )

    assert model is not None

    model = EmbeddingModel.from_pretrained_hf(
        WhichModel.Clip,
        model_id="openai/clip-vit-base-patch32",
    )

    assert model is not None


def test_clip_model_query(clip_model):

    data = embed_query(["Photo of a monkey?"], clip_model)
    assert len(data) == 1
    assert data[0].embedding is not None
    assert len(data[0].embedding) == 512


def test_clip_model_file(clip_model, test_image_file):

    data = embed_file(test_image_file, clip_model)

    assert data[0].embedding is not None
    assert len(data[0].embedding) == 512


def test_clip_model_directory(clip_model):

    data = embed_image_directory("test_files/clip", clip_model)
    assert len(data) == 5
    assert data[0].embedding is not None
    assert len(data[0].embedding) == 512
