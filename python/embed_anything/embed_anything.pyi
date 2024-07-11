"""
This module provides functions and classes for embedding queries, files, and directories using different embedding models.

The module includes the following functions:
- `embed_query`: Embeds the given query and returns an EmbedData object.
- `embed_file`: Embeds the file at the given path and returns a list of EmbedData objects.
- `embed_directory`: Embeds all the files in the given directory and returns a list of EmbedData objects.

The module also includes the `EmbedData` class, which represents the data of an embedded file.

Usage:
------
To embed a query, use the `embed_query` function:
    embed_query(query: list[str], embeder: str) -> list[EmbedData]

To embed a file, use the `embed_file` function:
    embed_file(file_path: str, embeder: str) -> list[EmbedData]

To embed a directory, use the `embed_directory` function:
    embed_directory(file_path: str, embeder: str) -> list[EmbedData]

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

from .embed_anything import *

def embed_query(query: list[str], embeder: str) -> list[EmbedData]:
    """Embeds the given query and returns an EmbedData object.

    Args:
        query: The query to embed.
        embeder: The name of the embedding model to use. Choose between "OpenAI" and "Bert"

    Returns:
        An EmbedData object.
    """

def embed_file(file_path: str, embeder: str) -> list[EmbedData]:
    """
    Embeds the file at the given path and returns a list of EmbedData objects.

    - Text -> "OpenAI", "Bert"
    - Image -> "Clip"
    - Audio -> "Whisper-Bert"

    Args:
        file_path: The path to the file to embed.
        embeder: The name of the embedding model to use.

    Returns:
        A list of EmbedData objects.

    """

def embed_directory(file_path: str, embeder: str) -> list[EmbedData]:
    """
    Embeds all the files in the given directory and returns a list of EmbedData objects.

    Args:
        file_path: The path to the directory containing the files to embed.
        embeder: The name of the embedding model to use. Choose between "OpenAI" and "Bert"

    Returns:
    - A list of EmbedData objects.
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
