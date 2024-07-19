import embed_anything
from embed_anything import JinaConfig, EmbedConfig, AudioDecoderConfig
import time

start_time = time.time()

# choose any whisper or distilwhisper model from https://huggingface.co/distil-whisper or https://huggingface.co/collections/openai/whisper-release-6501bba2cf999715fd953013
audio_decoder_config = AudioDecoderConfig(
    decoder_model_id="openai/whisper-tiny.en",
    decoder_revision="main",
    model_type="tiny-en",
    quantized=False,
)
jina_config = JinaConfig(
    model_id="jinaai/jina-embeddings-v2-small-en", revision="main", chunk_size=100
)

config = EmbedConfig(jina=jina_config, audio_decoder=audio_decoder_config)
data = embed_anything.embed_file(
    "test_files/audio/samples_hp0.wav", embeder="Audio", config=config
)
print(data[0].metadata)
end_time = time.time()
print("Time taken: ", end_time - start_time)
