import os
import re
import uuid
from typing import List, Dict
from abc import ABC, abstractmethod
from ._embed_anything import EmbedData


class Adapter(ABC):
    def __init__(self, api_key: str):
        self.api_key = api_key

    @abstractmethod
    def create_index(self, dimension: int, metric: str, index_name: str, **kwargs):
        pass

    @abstractmethod
    def delete_index(self, index_name: str):
        pass

    @abstractmethod
    def convert(self, embeddings: List[EmbedData]) -> List[Dict]:
        pass

    @abstractmethod
    def upsert(self, data: List[Dict]):
        data = self.convert(data)
        pass

