from .embed_anything import *
from embed_anything import EmbedData

def embed_file(file_path: str) -> list[EmbedData]:
    """
    Embeds the file at the given path and returns a list of EmbedData objects.
    """

class EmbedData:
    """
    Represents the data of an embedded file.
    """
    embedding: list[float]
    text: str