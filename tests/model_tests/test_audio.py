from embed_anything import AudioDecoderModel, EmbeddingModel, embed_audio_file
import pytest


def test_audio_decoder(audio_decoder: AudioDecoderModel):
    assert audio_decoder is not None


def test_audio_embed_file(
    audio_decoder: AudioDecoderModel, bert_model: EmbeddingModel, test_audio_file
):
    assert audio_decoder is not None
    assert bert_model is not None
    data = embed_audio_file(test_audio_file, audio_decoder, bert_model)
    assert data is not None
    assert len(data) == 1
    assert data[0].embedding is not None
    assert len(data[0].embedding) == 384
