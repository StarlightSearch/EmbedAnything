from typing import List
import pytest
from embed_anything import (
    Adapter,
    AudioDecoderModel,
    EmbedData,
    EmbeddingModel,
    WhichModel,
)


@pytest.fixture
def clip_model() -> EmbeddingModel:
    model = EmbeddingModel.from_pretrained_hf(
        WhichModel.Clip, model_id="openai/clip-vit-base-patch32", revision="refs/pr/15"
    )
    return model


@pytest.fixture
def test_files_directory() -> str:
    return "test_files"


@pytest.fixture
def test_pdf_file(test_files_directory) -> str:
    return f"{test_files_directory}/test.pdf"


@pytest.fixture
def test_txt_file(test_files_directory) -> str:
    return f"{test_files_directory}/test.txt"


@pytest.fixture
def test_image_file(test_files_directory) -> str:
    return f"{test_files_directory}/clip/monkey1.jpg"


@pytest.fixture
def test_audio_file(test_files_directory) -> str:
    return f"{test_files_directory}/audio/samples_jfk.wav"


@pytest.fixture
def test_image_directory(test_files_directory) -> str:
    return f"{test_files_directory}/clip"


@pytest.fixture
def test_text_directory(test_files_directory) -> str:
    return f"{test_files_directory}"


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


@pytest.fixture
def audio_decoder() -> AudioDecoderModel:
    model = AudioDecoderModel.from_pretrained_hf(
        model_id="openai/whisper-tiny", revision="main", model_type="tiny-en"
    )
    return model


class DummyAdapter(Adapter):

    def create_index(self, dimension: int, metric: str, index_name: str, **kwargs):
        pass

    def delete_index(self, index_name: str):
        pass

    def convert(self, data: List[EmbedData]) -> List[EmbedData]:
        return data

    def upsert(self, data: List[EmbedData]) -> None:
        data = self.convert(data)
        return 1


@pytest.fixture
def dummy_adapter() -> DummyAdapter:
    return DummyAdapter("dummy")
