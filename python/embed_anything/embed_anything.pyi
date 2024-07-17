"""
This module provides functions and classes for embedding queries, files, and directories using different embedding models.

The module includes the following functions:
- `embed_query`: Embeds the given query and returns an EmbedData object.
- `embed_file`: Embeds the file at the given path and returns a list of EmbedData objects.
- `embed_directory`: Embeds all the files in the given directory and returns a list of EmbedData objects.

The module also includes the `EmbedData` class, which represents the data of an embedded file.

The `EmbedData` class has the following attributes:
- `embedding`: The embedding of the file.
- `text`: The text for which the embedding is generated for.
- `metadata`: Additional metadata associated with the embedding.

Supported Embedding Models:
---------------------------
- Text Embedding Models:
    - "OpenAI"
    - "Bert"

- Image Embedding Models:
    - "Clip"

- Audio Embedding Models:
    - "Whisper-Bert"
"""

from typing import Optional
from .embed_anything import *

def embed_query(query: list[str], embeder: str) -> list[EmbedData]:
    """Embeds the given query and returns an EmbedData object.

    Args:
        query: The query to embed.
        embeder: The name of the embedding model to use. Choose between "OpenAI" and "Bert"

    Returns:
        An EmbedData object.
    """

def embed_file(
    file_path: str, embeder: str, config: Optional[EmbedConfig] = None
) -> list[EmbedData]:
    """
    Embeds the file at the given path and returns a list of EmbedData objects.

    - Text -> "OpenAI", "Bert"
    - Image -> "Clip"
    - Audio -> "Audio"

    Args:
        file_path: The path to the file to embed.
        embeder: The name of the embedding model to use.

    Returns:
        A list of EmbedData objects.

    Example:

    ```python
    import embed_anything
    data = embed_anything.embed_file("test_files/test.pdf", embeder="Bert
    ```

    """

def embed_directory(
    file_path: str, embeder: str, config: Optional[EmbedConfig]
) -> list[EmbedData]:
    """
    Embeds all the files in the given directory and returns a list of EmbedData objects.

    Args:
        file_path: The path to the directory containing the files to embed.
        embeder: The name of the embedding model to use. Choose between "OpenAI" and "Bert"
        config:  The configuration for the embedding model.

    Returns:
        A list of EmbedData objects.
    """

def embed_webpage(url: str, embeder: str) -> list[EmbedData]:
    """
    Embeds the webpage at the given URL and returns a list of EmbedData objects.

    Args:
        url: The URL of the webpage to embed.
        embeder: The name of the embedding model to use. Choose between "OpenAI", "Jina", "Bert"

    Returns:
        A list of EmbedData objects

    """

class EmbedData:
    """
    Represents the data of an embedded file.

    Attributes:
        embedding: The embedding of the file.
        text: The text for which the embedding is generated for.
        metadata: Additional metadata associated with the embedding.
    """

    embedding: list[float]
    text: str
    metadata: dict[str, str]

class BertConfig:
    """
    Represents the configuration for the Bert model.

    Attributes:
        model_id: The ID of the Bert model from huggingface.
        revision: The revision of the Bert model.
        chunk_size: The chunk size for the Bert model.
    """

    model_id: Optional[str] = None
    revision: Optional[str] = None
    chunk_size: Optional[int] = None

class JinaConfig:
    """
    Represents the configuration for the Jina model.

    Attributes:
        model_id: The ID of the Jina model from huggingface.
        revision: The revision of the Jina model.
        chunk_size: The chunk size for the Jina model.
    """

    model_id: Optional[str] = None
    revision: Optional[str] = None
    chunk_size: Optional[int] = None

class OpenAIConfig:
    """
    Represents the configuration for the OpenAI model.

    Attributes:
        model: The name of the OpenAI model.
        api_key: The API key for the OpenAI model.
        chunk_size: The chunk size for the OpenAI model.
    """

    model: Optional[str] = None
    api_key: Optional[str] = None
    chunk_size: Optional[int] = None

class ClipConfig:
    """
    Represents the configuration for the Clip model.

    Attributes:
        model_id: The ID of the Clip model from huggingface.
        revision: The revision of the Clip model.
    """

    model_id: Optional[str] = None
    revision: Optional[str] = None

class AudioDecoderConfig:
    """
    Represents the configuration for the Audio Decoder model.

    Attributes:
        decoder_model_id: The ID of the Audio Decoder model from huggingface.
        decoder_revision: The revision of the Audio Decoder model.
        model_type: The type of the Audio Decoder model.
        quantized: Whether the Audio Decoder model is quantized.
    """

    decoder_model_id: Optional[str] = None
    decoder_revision: Optional[str] = None
    model_type: Optional[str] = None
    quantized: Optional[bool] = None

class EmbedConfig:
    """
    Represents the configuration for the embedding model.

    Attributes:
        bert: The configuration for the Bert model.
        openai: The configuration for the OpenAI model.
        clip: The configuration for the Clip model.
    """

    bert: Optional[BertConfig] = None
    openai: Optional[OpenAIConfig] = None
    clip: Optional[ClipConfig] = None
