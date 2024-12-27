from enum import Enum
from typing import List, Dict
from abc import ABC, abstractmethod

class Adapter(ABC):
    def __init__(self, api_key: str):
        """
        Initializes the Adapter object.

        Args:
            api_key: The API key for accessing the adapter.
        """

    @abstractmethod
    def create_index(self, dimension: int, metric: str, index_name: str, **kwargs): ...
    """
    Creates an index for storing the embeddings.

    Args:
        dimension: The dimension of the embeddings.
        metric: The metric for measuring the distance between embeddings.
        index_name: The name of the index.
        kwargs: Additional keyword arguments.
    """
    @abstractmethod
    def delete_index(self, index_name: str):
        """
        Deletes an index.

        Args:
            index_name: The name of the index to delete.
        """

    @abstractmethod
    def convert(self, embeddings: List[List[EmbedData]]) -> List[Dict]:
        """
        Converts the embeddings to a list of dictionaries.

        Args:
            embeddings: The list of embeddings.

        Returns:
            A list of dictionaries.
        """

    @abstractmethod
    def upsert(self, data: List[Dict]):
        """
        Upserts the data into the index.

        Args:
            data: The list of data to upsert.
        """

def embed_query(
    query: list[str], embedder: EmbeddingModel, config: TextEmbedConfig | None = None
) -> list[EmbedData]:
    """
    Embeds the given query and returns a list of EmbedData objects.

    Args:
        query: The query to embed.
        embedder: The embedding model to use.
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
    embedder: EmbeddingModel,
    config: TextEmbedConfig | None = None,
    adapter: Adapter | None = None,
) -> list[EmbedData]:
    """
    Embeds the given file and returns a list of EmbedData objects.

    Args:
        file_path: The path to the file to embed.
        embedder: The embedding model to use.
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
    data = embed_anything.embed_file("test_files/test.pdf", embedder=model)
    ```
    """

def embed_directory(
    file_path: str,
    embedder: EmbeddingModel,
    extensions: list[str],
    config: TextEmbedConfig | None = None,
    adapter: Adapter | None = None,
) -> list[EmbedData]:
    """
    Embeds the files in the given directory and returns a list of EmbedData objects.

    Args:
        file_path: The path to the directory containing the files to embed.
        embedder: The embedding model to use.
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
    data = embed_anything.embed_directory("test_files", embedder=model, extensions=[".pdf"])
    ```
    """

def embed_image_directory(
    file_path: str,
    embedder: EmbeddingModel,
    config: ImageEmbedConfig | None = None,
    adapter: Adapter | None = None,
) -> list[EmbedData]:
    """
    Embeds the images in the given directory and returns a list of EmbedData objects.

    Args:
        file_path: The path to the directory containing the images to embed.
        embedder: The embedding model to use.
        config: The configuration for the embedding model.
        adapter: The adapter to use for storing the embeddings in a vector database.

    Returns:
        A list of EmbedData objects.
    """

def embed_webpage(
    url: str,
    embedder: EmbeddingModel,
    config: TextEmbedConfig | None,
    adapter: Adapter | None,
) -> list[EmbedData] | None:
    """Embeds the webpage at the given URL and returns a list of EmbedData
    objects.

    Args:
        url: The URL of the webpage to embed.
        embedder: The name of the embedding model to use. Choose between "OpenAI", "Jina", "Bert"
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
        "https://www.akshaymakes.com/", embedder="OpenAI", config=config
    )
    ```
    """

def embed_audio_file(
    file_path: str,
    audio_decoder: AudioDecoderModel,
    embedder: EmbeddingModel,
    text_embed_config: TextEmbedConfig | None = TextEmbedConfig(
        chunk_size=200, batch_size=32
    ),
) -> list[EmbedData]:
    """
    Embeds the given audio file and returns a list of EmbedData objects.

    Args:
        file_path: The path to the audio file to embed.
        audio_decoder: The audio decoder model to use.
        embedder: The embedding model to use.
        text_embed_config: The configuration for the embedding model.

    Returns:
        A list of EmbedData objects.

    Example:
    ```python

    import embed_anything
    audio_decoder = embed_anything.AudioDecoderModel.from_pretrained_hf(
        "openai/whisper-tiny.en", revision="main", model_type="tiny-en", quantized=False
    )

    embedder = embed_anything.EmbeddingModel.from_pretrained_hf(
        embed_anything.WhichModel.Bert,
        model_id="sentence-transformers/all-MiniLM-L6-v2",
        revision="main",
    )

    config = embed_anything.TextEmbedConfig(chunk_size=200, batch_size=32)
    data = embed_anything.embed_audio_file(
        "test_files/audio/samples_hp0.wav",
        audio_decoder=audio_decoder,
        embedder=embedder,
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

class ColpaliModel:
    """
    Represents the Colpali model.
    """

    def __init__(self, model_id: str, revision: str | None = None):
        """
        Initializes the ColpaliModel object.

        Args:
            model_id: The ID of the model from Hugging Face.
            revision: The revision of the model.
        """

    def from_pretrained(model_id: str, revision: str | None = None) -> ColpaliModel:
        """
        Loads a pre-trained Colpali model from the Hugging Face model hub.

        Args:
            model_id: The ID of the model from Hugging Face.
            revision: The revision of the model.

        Returns:
            A ColpaliModel object.
        """

    def from_pretrained_onnx(
        model_id: str, revision: str | None = None
    ) -> ColpaliModel:
        """
        Loads a pre-trained Colpali model from the Hugging Face model hub.

        Args:
            model_id: The ID of the model from Hugging Face.
            revision: The revision of the model.

        Returns:
            A ColpaliModel object.
        """

    def embed_file(self, file_path: str, batch_size: int | None = 1) -> list[EmbedData]:
        """
        Embeds the given pdf file and returns a list of EmbedData objects for each page in the file This first convert the pdf file into images and then embed each image.

        Args:
            file_path: The path to the pdf file to embed.
            batch_size: The batch size for processing the embeddings. Default is 1.

        Returns:
            A list of EmbedData objects for each page in the file.
        """

    def embed_query(self, query: str) -> list[EmbedData]:
        """
        Embeds the given query and returns a list of EmbedData objects.

        Args:
            query: The query to embed.

        Returns:
            A list of EmbedData objects.
        """

class JinaReranker:
    """
    Represents the Jina Reranker model.
    """

    def __init__(self, model_id: str, revision: str | None = None, dtype: Dtype | None = None):
        """
        Initializes the JinaReranker object.
        """

    def from_pretrained(model_id: str, revision: str | None = None, dtype: Dtype | None = None) -> JinaReranker:
        """
        Loads a pre-trained Jina Reranker model from the Hugging Face model hub.
        """

    def rerank(self, query: list[str], documents: list[str], top_k: int) -> RerankerResult:
        """
        Reranks the given documents for the query and returns a list of RerankerResult objects.
        """

class Dtype(Enum):
    FP16 = "FP16"
    INT8 = "INT8"
    Q4 = "Q4"
    UINT8 = "UINT8"
    BNB4 = "BNB4"

class RerankerResult:
    """
    Represents the result of the reranking process.
    """
    query: str
    documents: list[DocumentRank]

class DocumentRank:
    """
    Represents the rank of a document.
    """
    document: str
    relevance_score: float
    rank: int

class TextEmbedConfig:
    """
    Represents the configuration for the Text Embedding model.

    Attributes:
        chunk_size: The chunk size for the Text Embedding model.
        batch_size: The batch size for processing the embeddings. Default is 32. Based on the memory, you can increase or decrease the batch size.
        splitting_strategy: The strategy to use for splitting the text into chunks. Default is "sentence".
        semantic_encoder: The semantic encoder for the Text Embedding model. Default is None.
        use_ocr: A flag indicating whether to use OCR for the Text Embedding model. Default is False.
    """

    def __init__(
        self,
        chunk_size: int | None = 256,
        overlap_ratio: float | None = 0.0,
        batch_size: int | None = 32,
        buffer_size: int | None = 100,
        splitting_strategy: str | None = "sentence",
        semantic_encoder: EmbeddingModel | None = None,
        use_ocr: bool | None = False,
    ):
        self.chunk_size = chunk_size
        self.overlap_ratio = overlap_ratio
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.splitting_strategy = splitting_strategy
        self.semantic_encoder = semantic_encoder
        self.use_ocr = use_ocr
    chunk_size: int | None
    overlap_ratio: float | None
    batch_size: int | None
    buffer_size: int | None
    splitting_strategy: str | None
    semantic_encoder: EmbeddingModel | None
    use_ocr: bool | None

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
    """
    Loads an ONNX embedding model.

    Args:
        model_architecture (WhichModel): The architecture of the embedding model to use.
        model_id (str): The ID of the model.
        revision (str | None, optional): The revision of the model. Defaults to None.

    Returns:
        EmbeddingModel: An initialized EmbeddingModel object.

    Example:
    ```python
    model = EmbeddingModel.from_pretrained_onnx(
        model_architecture=WhichModel.Bert,
        model_id="BGESmallENV15Q"
    )
    ```

    Note:
    This method loads a pre-trained model in ONNX format, which can offer improved inference speed
    compared to standard PyTorch models. ONNX models are particularly useful for deployment
    scenarios where performance is critical.
    """
    def from_pretrained_onnx(
        model_architecture: WhichModel, model_id: str, revision: str | None = None
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
    Colpali = ("Colpali",)
    SparseBert = ("SparseBert",)

class ONNXModel(Enum):
    """
    Enum representing various ONNX models.

    ```markdown
    | Enum Variant                     | Description                                      |
    |----------------------------------|--------------------------------------------------|
    | `AllMiniLML6V2`                  | sentence-transformers/all-MiniLM-L6-v2           |
    | `AllMiniLML6V2Q`                 | Quantized sentence-transformers/all-MiniLM-L6-v2 |
    | `AllMiniLML12V2`                 | sentence-transformers/all-MiniLM-L12-v2          |
    | `AllMiniLML12V2Q`                | Quantized sentence-transformers/all-MiniLM-L12-v2|
    | `BGEBaseENV15`                   | BAAI/bge-base-en-v1.5                            |
    | `BGEBaseENV15Q`                  | Quantized BAAI/bge-base-en-v1.5                  |
    | `BGELargeENV15`                  | BAAI/bge-large-en-v1.5                           |
    | `BGELargeENV15Q`                 | Quantized BAAI/bge-large-en-v1.5                 |
    | `BGESmallENV15`                  | BAAI/bge-small-en-v1.5 - Default                 |
    | `BGESmallENV15Q`                 | Quantized BAAI/bge-small-en-v1.5                 |
    | `NomicEmbedTextV1`               | nomic-ai/nomic-embed-text-v1                     |
    | `NomicEmbedTextV15`              | nomic-ai/nomic-embed-text-v1.5                   |
    | `NomicEmbedTextV15Q`             | Quantized nomic-ai/nomic-embed-text-v1.5         |
    | `ParaphraseMLMiniLML12V2`        | sentence-transformers/paraphrase-MiniLM-L6-v2    |
    | `ParaphraseMLMiniLML12V2Q`       | Quantized sentence-transformers/paraphrase-MiniLM-L6-v2 |
    | `ParaphraseMLMpnetBaseV2`        | sentence-transformers/paraphrase-mpnet-base-v2   |
    | `BGESmallZHV15`                  | BAAI/bge-small-zh-v1.5                           |
    | `MultilingualE5Small`            | intfloat/multilingual-e5-small                   |
    | `MultilingualE5Base`             | intfloat/multilingual-e5-base                    |
    | `MultilingualE5Large`            | intfloat/multilingual-e5-large                   |
    | `MxbaiEmbedLargeV1`              | mixedbread-ai/mxbai-embed-large-v1               |
    | `MxbaiEmbedLargeV1Q`             | Quantized mixedbread-ai/mxbai-embed-large-v1     |
    | `GTEBaseENV15`                   | Alibaba-NLP/gte-base-en-v1.5                     |
    | `GTEBaseENV15Q`                  | Quantized Alibaba-NLP/gte-base-en-v1.5           |
    | `GTELargeENV15`                  | Alibaba-NLP/gte-large-en-v1.5                    |
    | `GTELargeENV15Q`                 | Quantized Alibaba-NLP/gte-large-en-v1.5          |
    | `JINAV2SMALLEN`                  | jinaai/jina-embeddings-v2-small-en               |
    | `JINAV2BASEEN`                   | jinaai/jina-embeddings-v2-base-en                |
    | `JINAV3`                         | jinaai/jina-embeddings-v3                        |
    | `SPLADEPPENV1`                   | prithivida/Splade_PP_en_v1                      |
    | `SPLADEPPENV2`                   | prithivida/Splade_PP_en_v2                      |
    ```
    """

    AllMiniLML6V2 = "AllMiniLML6V2"

    AllMiniLML6V2Q = "AllMiniLML6V2Q"

    AllMiniLML12V2 = "AllMiniLML12V2"

    AllMiniLML12V2Q = "AllMiniLML12V2Q"

    BGEBaseENV15 = "BGEBaseENV15"

    BGEBaseENV15Q = "BGEBaseENV15Q"

    BGELargeENV15 = "BGELargeENV15"

    BGELargeENV15Q = "BGELargeENV15Q"

    BGESmallENV15 = "BGESmallENV15"

    BGESmallENV15Q = "BGESmallENV15Q"

    NomicEmbedTextV1 = "NomicEmbedTextV1"

    NomicEmbedTextV15 = "NomicEmbedTextV15"

    NomicEmbedTextV15Q = "NomicEmbedTextV15Q"

    ParaphraseMLMiniLML12V2 = "ParaphraseMLMiniLML12V2"

    ParaphraseMLMiniLML12V2Q = "ParaphraseMLMiniLML12V2Q"

    ParaphraseMLMpnetBaseV2 = "ParaphraseMLMpnetBaseV2"

    BGESmallZHV15 = "BGESmallZHV15"

    MultilingualE5Small = "MultilingualE5Small"

    MultilingualE5Base = "MultilingualE5Base"

    MultilingualE5Large = "MultilingualE5Large"

    MxbaiEmbedLargeV1 = "MxbaiEmbedLargeV1"

    MxbaiEmbedLargeV1Q = "MxbaiEmbedLargeV1Q"

    GTEBaseENV15 = "GTEBaseENV15"

    GTEBaseENV15Q = "GTEBaseENV15Q"

    GTELargeENV15 = "GTELargeENV15"

    GTELargeENV15Q = "GTELargeENV15Q"

    JINAV2SMALLEN = "JINAV2SMALLEN"

    JINAV2BASEEN = "JINAV2BASEEN"

    JINAV3 = "JINAV3"

    SPLADEPPENV1 = "SPLADEPPENV1"

    SPLADEPPENV2 = "SPLADEPPENV2"
