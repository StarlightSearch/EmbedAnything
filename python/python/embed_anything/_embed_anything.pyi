from enum import Enum
from typing import List, Dict
from abc import ABC, abstractmethod

class Adapter(ABC):
    def __init__(self, api_key: str): ...
    @abstractmethod
    def create_index(self, dimension: int, metric: str, index_name: str, **kwargs): ...
    @abstractmethod
    def delete_index(self, index_name: str): ...
    @abstractmethod
    def convert(self, embeddings: List[List[EmbedData]]) -> List[Dict]: ...
    @abstractmethod
    def upsert(self, data: List[Dict]): ...

def embed_query(
    query: list[str], embeder: EmbeddingModel, config: TextEmbedConfig | None = None
) -> list[EmbedData]:
    """
    Embeds the given query and returns a list of EmbedData objects.

    Args:
        query: The query to embed.
        embeder: The embedding model to use.
        config: The configuration for the embedding model.

    Returns:
        A list of EmbedData objects.

    Example:

    ```python
    import embed_anything
    model = embed_anything.EmbeddingModel.from_pretrained_hf(
        embed_anything.WhichModel.Bert,
        model_id="sentence-transformers/all-MiniLM-L6-v2",
        revision="main",
    )
    ```
    """

def embed_file(
    file_path: str,
    embeder: EmbeddingModel,
    config: TextEmbedConfig | None = None,
    adapter: Adapter | None = None,
) -> list[EmbedData]:
    """
    Embeds the given file and returns a list of EmbedData objects.

    Args:
        file_path: The path to the file to embed.
        embeder: The embedding model to use.
        config: The configuration for the embedding model.
        adapter: The adapter to use for storing the embeddings in a vector database.

    Returns:
        A list of EmbedData objects.

    Example:
    ```python
    import embed_anything
    model = embed_anything.EmbeddingModel.from_pretrained_hf(
        embed_anything.WhichModel.Bert,
        model_id="sentence-transformers/all-MiniLM-L6-v2",
        revision="main",
    )
    data = embed_anything.embed_file("test_files/test.pdf", embeder=model)
    ```
    """

def embed_directory(
    file_path: str,
    embeder: EmbeddingModel,
    extensions: list[str],
    config: TextEmbedConfig | None = None,
    adapter: Adapter | None = None,
) -> list[EmbedData]:
    """
    Embeds the files in the given directory and returns a list of EmbedData objects.

    Args:
        file_path: The path to the directory containing the files to embed.
        embeder: The embedding model to use.
        extensions: The list of file extensions to consider for embedding.
        config: The configuration for the embedding model.
        adapter: The adapter to use for storing the embeddings in a vector database.

    Returns:
        A list of EmbedData objects.

    Example:
    ```python
    import embed_anything
    model = embed_anything.EmbeddingModel.from_pretrained_hf(
        embed_anything.WhichModel.Bert,
        model_id="sentence-transformers/all-MiniLM-L6-v2",
        revision="main",
    )
    data = embed_anything.embed_directory("test_files", embeder=model, extensions=[".pdf"])
    ```
    """

def embed_image_directory(
    file_path: str,
    embeder: EmbeddingModel,
    config: ImageEmbedConfig | None = None,
    adapter: Adapter | None = None,
) -> list[EmbedData]:
    """
    Embeds the images in the given directory and returns a list of EmbedData objects.

    Args:
        file_path: The path to the directory containing the images to embed.
        embeder: The embedding model to use.
        config: The configuration for the embedding model.
        adapter: The adapter to use for storing the embeddings in a vector database.

    Returns:
        A list of EmbedData objects.
    """

def embed_webpage(
    url: str,
    embeder: EmbeddingModel,
    config: TextEmbedConfig | None,
    adapter: Adapter | None,
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

def embed_audio_file(
    file_path: str,
    audio_decoder: AudioDecoderModel,
    embeder: EmbeddingModel,
    text_embed_config: TextEmbedConfig | None = TextEmbedConfig(
        chunk_size=200, batch_size=32
    ),
) -> list[EmbedData]:
    """
    Embeds the given audio file and returns a list of EmbedData objects.

    Args:
        file_path: The path to the audio file to embed.
        audio_decoder: The audio decoder model to use.
        embeder: The embedding model to use.
        text_embed_config: The configuration for the embedding model.

    Returns:
        A list of EmbedData objects.

    Example:
    ```python

    import embed_anything
    audio_decoder = embed_anything.AudioDecoderModel.from_pretrained_hf(
        "openai/whisper-tiny.en", revision="main", model_type="tiny-en", quantized=False
    )

    embeder = embed_anything.EmbeddingModel.from_pretrained_hf(
        embed_anything.WhichModel.Bert,
        model_id="sentence-transformers/all-MiniLM-L6-v2",
        revision="main",
    )

    config = embed_anything.TextEmbedConfig(chunk_size=200, batch_size=32)
    data = embed_anything.embed_audio_file(
        "test_files/audio/samples_hp0.wav",
        audio_decoder=audio_decoder,
        embeder=embeder,
        text_embed_config=config,
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

class TextEmbedConfig:
    """
    Represents the configuration for the Text Embedding model.

    Attributes:
        chunk_size: The chunk size for the Text Embedding model.
        batch_size: The batch size for processing the embeddings. Default is 32. Based on the memory, you can increase or decrease the batch size.
    """

    def __init__(self, chunk_size: int | None = None, batch_size: int | None = None):
        self.chunk_size = chunk_size
        self.batch_size = batch_size
    chunk_size: int | None
    batch_size: int | None

class ImageEmbedConfig:
    """
    Represents the configuration for the Image Embedding model.

    Attributes:
        buffer_size: The buffer size for the Image Embedding model. Default is 100.
    """

    def __init__(self, buffer_size: int | None = None):
        self.buffer_size = buffer_size
    buffer_size: int | None

class EmbeddingModel:
    """
    Represents an embedding model.
    """

    """
    Loads an embedding model from the Hugging Face model hub.

    Args:
        model_id: The ID of the model.
        revision: The revision of the model.
    
    Returns:
        An EmbeddingModel object.

    Example:
    ```python
    model = EmbeddingModel.from_pretrained_hf(
        model_id="prithivida/miniMiracle_te_v1",
        revision="main"
    )
    ```

    """
    def from_pretrained_hf(
        model: WhichModel, model_id: str, revision: str | None = None
    ) -> EmbeddingModel: ...

    """
    Loads an embedding model from a cloud-based service.

    Args:
        model (WhichModel): The cloud service to use. Currently supports WhichModel.OpenAI and WhichModel.Cohere.
        model_id (str): The ID of the model to use.
            - For OpenAI, see available models at https://platform.openai.com/docs/guides/embeddings/embedding-models
            - For Cohere, see available models at https://docs.cohere.com/docs/cohere-embed
        api_key (str | None, optional): The API key for accessing the model. If not provided, it is taken from the environment variable:
            - For OpenAI: OPENAI_API_KEY
            - For Cohere: CO_API_KEY

    Returns:
        EmbeddingModel: An initialized EmbeddingModel object.

    Raises:
        ValueError: If an unsupported model is specified.

    Example:
    ```python
    # Using Cohere
    model = EmbeddingModel.from_pretrained_cloud(
        model=WhichModel.Cohere, 
        model_id="embed-english-v3.0"
    )

    # Using OpenAI
    model = EmbeddingModel.from_pretrained_cloud(
        model=WhichModel.OpenAI, 
        model_id="text-embedding-3-small"
    )
    ```
    """
    def from_pretrained_cloud(
        model: WhichModel, model_id: str, api_key: str | None = None
    ) -> EmbeddingModel: ...

class AudioDecoderModel:
    """
    Represents an audio decoder model.

    Attributes:
        model_id: The ID of the audio decoder model.
        revision: The revision of the audio decoder model.
        model_type: The type of the audio decoder model.
        quantized: A flag indicating whether the audio decoder model is quantized or not.

    Example:
    ```python

    model = embed_anything.AudioDecoderModel.from_pretrained_hf(
        model_id="openai/whisper-tiny.en",
        revision="main",
        model_type="tiny-en",
        quantized=False
    )
    ```
    """

    model_id: str
    revision: str
    model_type: str
    quantized: bool

    def from_pretrained_hf(
        model_id: str | None = None,
        revision: str | None = None,
        model_type: str | None = None,
        quantized: bool | None = None,
    ): ...

class WhichModel(Enum):
    OpenAI = ("OpenAI",)
    Cohere = ("Cohere",)
    Bert = ("Bert",)
    Jina = ("Jina",)
    Clip = ("Clip",)
