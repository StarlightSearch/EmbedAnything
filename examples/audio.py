import embed_anything
from embed_anything import (
    AudioDecoderModel,
    EmbeddingModel,
    embed_audio_file,
    TextEmbedConfig,
)
import time

start_time = time.time()

# choose any whisper or distilwhisper model from https://huggingface.co/distil-whisper or https://huggingface.co/collections/openai/whisper-release-6501bba2cf999715fd953013
audio_decoder = AudioDecoderModel.from_pretrained_hf(
    "openai/whisper-tiny.en", revision="main", model_type="tiny-en", quantized=False
)

embedder = EmbeddingModel.from_pretrained_hf(
    embed_anything.WhichModel.Bert,
    model_id="sentence-transformers/all-MiniLM-L6-v2",
    revision="main",
)

config = TextEmbedConfig(chunk_size=200, batch_size=32)
data = embed_anything.embed_audio_file(
    "test_files/audio/samples_hp0.wav",
    audio_decoder=audio_decoder,
    embedder=embedder,
    text_embed_config=config,
)
print(data[0].metadata)
end_time = time.time()
print("Time taken: ", end_time - start_time)
