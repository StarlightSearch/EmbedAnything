import pytest
from embed_anything import EmbeddingModel, WhichModel


@pytest.fixture
def clip_model() -> EmbeddingModel:
    model = EmbeddingModel.from_pretrained_hf(
        WhichModel.Clip, model_id="openai/clip-vit-base-patch32", revision="refs/pr/15"
    )
    return model


@pytest.fixture
def jina_model() -> EmbeddingModel:
    model = EmbeddingModel.from_pretrained_hf(
        WhichModel.Jina, model_id="jinaai/jina-embeddings-v2-small-en", revision="main"
    )
    return model


@pytest.fixture
def bert_model() -> EmbeddingModel:
    model = EmbeddingModel.from_pretrained_hf(
        WhichModel.Bert,
        model_id="sentence-transformers/all-MiniLM-L6-v2",
        revision="main",
    )
    return model
