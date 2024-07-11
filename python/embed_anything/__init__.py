"""
This module provides functions and classes for embedding queries, files, and directories using different embedding models.

The module includes the following functions:

 - `embed_query`: Embeds the given query and returns an EmbedData object.
 - `embed_file`: Embeds the file at the given path and returns a list of EmbedData objects.
 - `embed_directory`: Embeds all the files in the given directory and returns a list of EmbedData objects.

The module also includes the `EmbedData` class, which represents the data of an embedded file.

Usage:
------
- To embed a query, use the `embed_query` function: \n
    `embed_query(query: list[str], embeder: str) -> list[EmbedData]`

- To embed a file, use the `embed_file` function: \n
    `embed_file(file_path: str, embeder: str) -> list[EmbedData]`

- To embed a directory, use the `embed_directory` function: \n
    `embed_directory(file_path: str, embeder: str) -> list[EmbedData]`

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

__doc__ = embed_anything.__doc__
if hasattr(embed_anything, "__all__"):
    __all__ = embed_anything.__all__
