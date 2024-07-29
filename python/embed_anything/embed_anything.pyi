from .embed_anything import *
from embed_anything.vectordb import Adapter

def embed_query(
    query: list[str], embeder: str, config: EmbedConfig | None = None
) -> list[EmbedData]:
    """Embeds the given query and returns an EmbedData object.

    Args:
        query: The query to embed.
        embeder: The name of the embedding model to use. Choose between "OpenAI", "Jina", "Bert"
        config: The configuration for the embedding model.

    Returns:
        An EmbedData object.

    Example:
    ```python
    import embed_anything
    config = embed_anything.EmbedConfig(
        bert=embed_anything.BertConfig(model_id="sentence-transformers/allMiniLM-L6-v2"),
    )
    data = embed_anything.embed_query(["Hello World", "This is a query"], embeder="Bert, config = config)
    ```
    """

def embed_file(
    file_path: str,
    embeder: str,
    config: EmbedConfig | None = None,
    adapter: Adapter | None = None,
) -> list[EmbedData]:
    """Embeds the file at the given path and returns a list of EmbedData
    objects.

    - Text -> "OpenAI", "Bert"
    - Image -> "Clip"
    - Audio -> "Audio"

    Args:
        file_path: The path to the file to embed.
        embeder: The name of the embedding model to use.
        config: The configuration for the embedding model.
        adapter: The adapter to use for storing the embeddings

    Returns:
        A list of EmbedData objects.

    Example:
    ```python
    import embed_anything
    config = embed_anything.EmbedConfig(
                    openai_config=embed_anything.OpenAIConfig(
                    model="text-embedding-3-small")
                    )
    data = embed_anything.embed_file("test_files/test.pdf",
                                    embeder="OpenAI,
                                    config = config)
    ```
    """

def embed_directory(
    file_path: str, embeder: str, config: EmbedConfig | None = None
) -> list[EmbedData]:
    """Embeds all the files in the given directory and returns a list of
    EmbedData objects.

    Args:
        file_path: The path to the directory containing the files to embed.
        embeder: The name of the embedding model to use. Choose between "OpenAI", "Jina", "Bert"
        config:  The configuration for the embedding model.
        adapter: The adapter to use for storing the embeddings in a vector database.

    Returns:
        A list of EmbedData objects.

    Example:
    ```python
    import embed_anything
    config = embed_anything.EmbedConfig(
                    bert_config = embed_anything.BertConfig(
                    model_id="sentence-transformers/allMiniLM-L6-v2")
                    )
    data = embed_anything.embed_directory("test_files", embeder="Bert", config=config)
    ```
    """

def embed_webpage(
    url: str, embeder: str, config: EmbedConfig | None, adapter: Adapter | None
) -> list[EmbedData] | None:
    """Embeds the webpage at the given URL and returns a list of EmbedData
    objects.

    Args:
        url: The URL of the webpage to embed.
        embeder: The name of the embedding model to use. Choose between "OpenAI", "Jina", "Bert"
        config: The configuration for the embedding model.
        adapter: The adapter to use for storing the embeddings.

    Returns:
        A list of EmbedData objects

    Example:
    ```python
    import embed_anything

    config = embed_anything.EmbedConfig(
        openai_config=embed_anything.OpenAIConfig(model="text-embedding-3-small")
    )
    data = embed_anything.embed_webpage(
        "https://www.akshaymakes.com/", embeder="OpenAI", config=config
    )
    ```
    """

class EmbedData:
    """Represents the data of an embedded file.

    Attributes:
        embedding: The embedding of the file.
        text: The text for which the embedding is generated for.
        metadata: Additional metadata associated with the embedding.
    """

    def __init__(self, embedding: list[float], text: str, metadata: dict[str, str]):
        self.embedding = embedding
        self.text = text
        self.metadata = metadata
    embedding: list[float]
    text: str
    metadata: dict[str, str]

class EmbedConfig:
    """Represents the configuration for the embedding model. Each config object
    should hold only one of the embedding model. Do not provide multiple.

    Attributes:
        bert: The configuration for the Bert model.
        openai: The configuration for the OpenAI model.
        clip: The configuration for the Clip model.
    """

    def __init__(
        self,
        bert: BertConfig | None = None,
        openai: OpenAIConfig | None = None,
        clip: ClipConfig | None = None,
        audio_decoder: AudioDecoderConfig | None = None,
    ):
        self.bert = bert
        self.openai = openai
        self.clip = clip
        self.audio_decoder = audio_decoder
    bert: BertConfig | None
    openai: OpenAIConfig | None
    clip: ClipConfig | None
    audio_decoder: AudioDecoderConfig | None

class BertConfig:
    """Represents the configuration for the Bert model.

    Attributes:
        model_id: The ID of the Bert model from huggingface.
        revision: The revision of the Bert model.
        chunk_size: The chunk size for the Bert model.
    """

    def __init__(
        self,
        model_id: str | None = None,
        revision: str | None = None,
        chunk_size: int | None = None,
    ):
        self.model_id = model_id
        self.revision = revision
        self.chunk_size = chunk_size
    model_id: str | None
    revision: str | None
    chunk_size: int | None

class JinaConfig:
    """Represents the configuration for the Jina model.

    Attributes:
        model_id: The ID of the Jina model from huggingface.
        revision: The revision of the Jina model.
        chunk_size: The chunk size for the Jina model.
    """

    def __init__(
        self,
        model_id: str | None = None,
        revision: str | None = None,
        chunk_size: int | None = None,
    ):
        self.model_id = model_id
        self.revision = revision
        self.chunk_size = chunk_size
    model_id: str | None = None
    revision: str | None = None
    chunk_size: int | None = None

class OpenAIConfig:
    """Represents the configuration for the OpenAI model.

    Attributes:
        model: The name of the OpenAI model.
        api_key: The API key for the OpenAI model.
        chunk_size: The chunk size for the OpenAI model.
    """

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        chunk_size: int | None = None,
    ):
        self.model = model
        self.api_key = api_key
        self.chunk_size = chunk_size
    model: str | None
    api_key: str | None
    chunk_size: int | None

class ClipConfig:
    """Represents the configuration for the Clip model.

    Attributes:
        model_id: The ID of the Clip model from huggingface.
        revision: The revision of the Clip model.
    """

    def __init__(self, model_id: str | None = None, revision: str | None = None):
        self.model_id = model_id
        self.revision = revision
    model_id: str | None
    revision: str | None

class AudioDecoderConfig:
    """
    Represents the configuration for the Audio Decoder model. Choose any whisper or
    distilwhisper model from https://huggingface.co/distil-whisper
    or https://huggingface.co/collections/openai/whisper-release-6501bba2cf999715fd953013

    Attributes:
        decoder_model_id: The ID of the Audio Decoder model from huggingface.
        decoder_revision: The revision of the Audio Decoder model.
        model_type: The type of the Audio Decoder model.
        quantized: Whether the Audio Decoder model is quantized.
    """

    def __init__(
        self,
        decoder_model_id: str | None = None,
        decoder_revision: str | None = None,
        model_type: str | None = None,
        quantized: bool | None = None,
    ):
        self.decoder_model_id = decoder_model_id
        self.decoder_revision = decoder_revision
        self.model_type = model_type
        self.quantized = quantized
    decoder_model_id: str | None
    decoder_revision: str | None
    model_type: str | None
    quantized: bool | None
