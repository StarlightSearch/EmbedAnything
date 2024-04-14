from .embed_anything import *

def embed_query(query: list[str], embeder: str) -> list[EmbedData]:
    """
    Embeds the given query and returns an EmbedData object.

    ### Arguments:
    - `query`: The query to embed.
    - `embeder`: The name of the embedding model to use. Choose between "OpenAI" and "AllMiniLmL12V2"

    ### Returns:
    - An EmbedData object.
    """
def embed_file(file_path: str, embeder: str) -> list[EmbedData]:
    """
    Embeds the file at the given path and returns a list of EmbedData objects.

    ### Arguments:
    - `file_path`: The path to the file to embed.
    - `embeder`: The name of the embedding model to use. Choose between "OpenAI" and "AllMiniLmL12V2"

    ### Returns:
    - A list of EmbedData objects.
    """

def embed_directory(file_path: str, embeder: str) -> list[EmbedData]:
    """
    Embeds all the files in the given directory and returns a list of EmbedData objects.

    ### Arguments:
    - `file_path`: The path to the directory containing the files to embed.
    - `embeder`: The name of the embedding model to use. Choose between "OpenAI" and "AllMiniLmL12V2"

    ### Returns:
    - A list of EmbedData objects.
    """

class EmbedData:
    """
    Represents the data of an embedded file.

    ### Attributes:
    - `embedding`: The embedding of the file.
    - `text`: The text for which the embedding is generated for.
    """
    embedding: list[float]
    text: str