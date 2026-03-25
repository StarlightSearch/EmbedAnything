from embed_anything import TextEmbedConfig, embed_directory, embed_file, embed_query
import pytest


def test_minimax_model_query(minimax_model):
    data = embed_query(["Hello world"], minimax_model)
    assert data[0].embedding is not None
    assert len(data[0].embedding) == 1536


def test_minimax_model_file(minimax_model, test_pdf_file):
    data = embed_file(test_pdf_file, minimax_model)
    assert data[0].embedding is not None
    assert len(data[0].embedding) == 1536


@pytest.mark.parametrize(
    "config", [TextEmbedConfig(batch_size=512, chunk_size=1000, buffer_size=512)]
)
def test_minimax_model_directory(minimax_model, config, test_files_directory):
    data = embed_directory(test_files_directory, minimax_model, config=config)
    assert data[0].embedding is not None
    assert len(data[0].embedding) == 1536
