from enum import Enum
from typing import List, Dict, Optional
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

def embed_files_batch(
    files: list[str],
    embedder: EmbeddingModel,
    config: TextEmbedConfig | None = None,
    adapter: Adapter | None = None,
) -> list[EmbedData]:
    """
    Embeds the given files and returns a list of EmbedData objects.

    Args:
        files: The list of files to embed.
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
    data = embed_anything.embed_files_batch(
        ["test_files/test.pdf", "test_files/test.txt"],
        embedder=model,
        config=embed_anything.TextEmbedConfig(),
        adapter=None,
    )
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

def embed_html(
    file_name: str,
    embedder: EmbeddingModel,
    origin: str | None = None,
    config: TextEmbedConfig | None = None,
    adapter: Adapter | None = None,
) -> list[EmbedData]:
    """
    Embeds the given HTML file and returns a list of EmbedData objects.

    Args:
        file_name: The path to the HTML file to embed.
        embedder: The embedding model to use.
        origin: The origin of the HTML file.
        config: The configuration for the embedding model.
        adapter: The adapter to use for storing the embeddings.

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
    data = embed_anything.embed_html(
        "test_files/test.html", embedder=model, origin="https://www.akshaymakes.com/"
    )
    ```
    """

def embed_audio_file(
    file_path: str,
    audio_decoder: AudioDecoderModel,
    embedder: EmbeddingModel,
    text_embed_config: TextEmbedConfig | None = TextEmbedConfig(
        chunk_size=1000, batch_size=32
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

    config = embed_anything.TextEmbedConfig(chunk_size=1000, batch_size=32)
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

class ColbertModel:
    """
    Represents the Colbert model.
    """

    def __init__(
        self,
        hf_model_id: str | None = None,
        revision: str | None = None,
        path_in_repo: str | None = None,
    ):
        """
        Initializes the ColbertModel object.
        """

    def from_pretrained_onnx(
        self,
        hf_model_id: str | None = None,
        revision: str | None = None,
        path_in_repo: str | None = None,
    ) -> ColbertModel:
        """
        Loads a pre-trained Colbert model from the Hugging Face model hub.

        Attributes:
            hf_model_id: The ID of the model from Hugging Face.
            revision: The revision of the model.
            path_in_repo: The path to the model in the repository.

        Returns:
            A ColbertModel object.
        """

    def embed(
        self, text_batch: list[str], batch_size: int | None = None, is_doc: bool = True
    ) -> list[EmbedData]:
        """
        Embeds the given text and returns a list of EmbedData objects.
        """

class Reranker:
    """
    Represents the Reranker model.
    """

    def __init__(
        self, model_id: str, revision: str | None = None, dtype: Dtype | None = None, path_in_repo: str | None = None
    ):
        """
        Initializes the Reranker object.
        """

    def from_pretrained(
        model_id: str, revision: str | None = None, dtype: Dtype | None = None, path_in_repo: str | None = None
    ) -> Reranker:
        """
        Loads a pre-trained Reranker model from the Hugging Face model hub.

        Args:
            model_id: The ID of the model from Hugging Face.
            revision: The revision of the model.
            dtype: The dtype of the model.
            path_in_repo: The path to the model in the repository.

        """

    def rerank(
        self, query: list[str], documents: list[str], batch_size: int
    ) -> RerankerResult:
        """
        Reranks the given documents for the query and returns a list of RerankerResult objects.

        Args:
            query: The query to rerank.
            documents: The list of documents to rerank.
            batch_size: The number of documents to process per batch.

        Returns:
            A list of RerankerResult objects.
        """

    def compute_scores(
        self, query: list[str], documents: list[str], batch_size: int
    ) -> list[list[float]]:
        """
        Computes the scores for the given query and documents.

        Args:
            query: The query to compute the scores for.
            documents: The list of documents to compute the scores for.
            batch_size: The batch size for processing the scores.

        Returns:
            A list of scores for the given query and documents.
        """
class Dtype(Enum):
    """
    Represents the data type of the model.
    """

    F16 = "F16"
    INT8 = "INT8"
    Q4 = "Q4"
    UINT8 = "UINT8"
    BNB4 = "BNB4"
    Q4F16 = "Q4F16"
    BF16 = "BF16"
    F32 = "F32"

class RerankerResult:
    """
    Represents the result of the reranking process.

    Attributes:
        query: The query to rerank.
        documents: The list of documents to rerank.
    """

    query: str
    documents: list[DocumentRank]

class DocumentRank:
    """
    Represents the rank of a document.

    Attributes:
        document: The document to rank.
        relevance_score: The relevance score of the document.
        rank: The rank of the document.
    """

    document: str
    relevance_score: float
    rank: int

class TextEmbedConfig:
    """
    Represents the configuration for the Text Embedding model.

    Attributes:
        chunk_size: The chunk size for the Text Embedding model. Default is 1000 Characters.
        batch_size: The batch size for processing the embeddings. Default is 32. Based on the memory, you can increase or decrease the batch size.
        buffer_size: The buffer size for the Text Embedding model. Default is 100.
        late_chunking: A flag indicating whether to use late chunking for the Text Embedding model. Use late chunking to increase the context that is taken into account for each chunk.  Default is False.
        splitting_strategy: The strategy to use for splitting the text into chunks. Default is "sentence". If semantic splitting is used, semantic_encoder is required.
        semantic_encoder: The semantic encoder for the Text Embedding model. Default is None.
        use_ocr: A flag indicating whether to use OCR for the Text Embedding model. Default is False.
        tesseract_path: The path to the Tesseract OCR executable. Default is None and uses the system path.
        pdf_backend: The backend to use for PDF text extraction. Currently only `lopdf` is supported. Default is `lopdf`.
    """

    def __init__(
        self,
        chunk_size: int | None = 1000,
        overlap_ratio: float | None = 0.0,
        batch_size: int | None = 32,
        late_chunking: bool | None = False,
        buffer_size: int | None = 100,
        splitting_strategy: str | None = "sentence",
        semantic_encoder: EmbeddingModel | None = None,
        use_ocr: bool | None = False,
        tesseract_path: str | None = None,
        pdf_backend: str | None = "lopdf",
    ):
        self.chunk_size = chunk_size
        self.overlap_ratio = overlap_ratio
        self.batch_size = batch_size
        self.late_chunking = late_chunking
        self.buffer_size = buffer_size
        self.splitting_strategy = splitting_strategy
        self.semantic_encoder = semantic_encoder
        self.use_ocr = use_ocr
        self.tesseract_path = tesseract_path
        self.pdf_backend = pdf_backend
    chunk_size: int | None
    overlap_ratio: float | None
    batch_size: int | None
    late_chunking: bool | None
    buffer_size: int | None
    splitting_strategy: str | None
    semantic_encoder: EmbeddingModel | None
    use_ocr: bool | None
    tesseract_path: str | None
    pdf_backend: str | None

class ImageEmbedConfig:
    """
    Represents the configuration for the Image Embedding model.

    Attributes:
        buffer_size: The buffer size for the Image Embedding model. Default is 100.
        batch_size: The batch size for processing the embeddings. Default is 32. Based on the memory, you can increase or decrease the batch size.
    """

    def __init__(self, buffer_size: int | None = None, batch_size: int | None = None):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
    buffer_size: int | None
    batch_size: int | None

class EmbeddingModel:
    """
    Represents an embedding model.
    """

    def from_pretrained_hf(
        model_id: str,
        revision: str | None = None,
        token: str | None = None,
        dtype: Dtype | None = None,
    ) -> EmbeddingModel:
        """
        Loads an embedding model from the Hugging Face model hub.

        Attributes:
            model_id: The ID of the model.
            revision: The revision of the model.
            token: The Hugging Face token.
            dtype: The dtype of the model.
        Returns:
            An EmbeddingModel object.

        Example:
        ```python
        model = EmbeddingModel.from_pretrained_hf(
            model_id="sentence-transformers/all-MiniLM-L6-v2",
            revision="main"
        )
        ```

        """

    def from_pretrained_cloud(
        model: WhichModel, model_id: str, api_key: str | None = None
    ) -> EmbeddingModel:
        """
        Loads an embedding model from a cloud-based service.

        Attributes:
            model (WhichModel): The cloud service to use. Currently supports WhichModel.OpenAI and WhichModel.Cohere.
            model_id (str): The ID of the model to use.

                - For OpenAI, see available models at https://platform.openai.com/docs/guides/embeddings/embedding-models
                - For Cohere, see available models at https://docs.cohere.com/docs/cohere-embed
                - For CohereVision, see available models at https://docs.cohere.com/docs/cohere-embed
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

    def from_pretrained_onnx(
        model: WhichModel,
        model_name: Optional[ONNXModel] | None = None,
        hf_model_id: Optional[str] | None = None,
        revision: Optional[str] | None = None,
        dtype: Optional[Dtype] | None = None,
        path_in_repo: Optional[str] | None = None,
    ) -> EmbeddingModel:
        """
        Loads an ONNX embedding model.

        Args:
            model (WhichModel): The architecture of the embedding model to use.
            model_name (ONNXModel | None, optional): The name of the model. Defaults to None.
            hf_model_id (str | None, optional): The ID of the model from Hugging Face. Defaults to None.
            revision (str | None, optional): The revision of the model. Defaults to None.
            dtype (Dtype | None, optional): The dtype of the model. Defaults to None.
            path_in_repo (str | None, optional): The path to the model in the repository. Defaults to None.
        Returns:
            EmbeddingModel: An initialized EmbeddingModel object.

        Atleast one of the following arguments must be provided:
            - model_name
            - hf_model_id

        If hf_model_id is provided, dtype is ignored and the path_in_repo has to be provided pointing to the model file in the repository.
        If model_name is provided, dtype is used to determine the model file to load.

        Example:
        ```python
        model = EmbeddingModel.from_pretrained_onnx(
            model=WhichModel.Bert,
            model_name=ONNXModel.BGESmallENV15Q,
            dtype=Dtype.Q4F16
        )

        model = EmbeddingModel.from_pretrained_onnx(
            model=WhichModel.Bert,
            hf_model_id="jinaai/jina-embeddings-v3",
            path_in_repo="onnx/model_fp16.onnx"
        )
        ```

        Note:
        This method loads a pre-trained model in ONNX format, which can offer improved inference speed
        compared to standard PyTorch models. ONNX models are particularly useful for deployment
        scenarios where performance is critical.
        """

    def embed_file(
        self,
        file_path: str,
        config: TextEmbedConfig | None = None,
        adapter: Adapter | None = None,
    ) -> list[EmbedData]:
        """
        Embeds the given file and returns a list of EmbedData objects.

        Args:
            file_path: The path to the file to embed.
            config: The configuration for the embedding.
            adapter: The adapter for the embedding.

        Returns:
            A list of EmbedData objects.
        """

    def embed_files_batch(
        self,
        files: list[str],
        config: TextEmbedConfig | None = None,
        adapter: Adapter | None = None,
    ) -> list[EmbedData]:
        """
        Embeds the given files and returns a list of EmbedData objects.

        Args:
            files: The list of files to embed.
            config: The configuration for the embedding.
            adapter: The adapter for the embedding.

        Returns:
            A list of EmbedData objects.
        """

    def embed_audio_file(
        self,
        audio_file: str,
        audio_decoder: AudioDecoderModel,
        config: TextEmbedConfig | None = None,
    ) -> list[EmbedData]:
        """
        Embeds the given audio file and returns a list of EmbedData objects.

        Args:
            audio_file: The path to the audio file to embed.
            audio_decoder: The audio decoder for the audio file.
            config: The configuration for the embedding.

        Returns:
            A list of EmbedData objects.
        """

    def embed_query(
        self,
        query: list[str],
        config: TextEmbedConfig | None = None,
    ) -> list[EmbedData]:
        """
        Embeds the given list of queries and returns a list of EmbedData objects.

        Args:
            query: The list of queries to embed.
            config: The configuration for the embedding.

        Returns:
            A list of EmbedData objects.
        """

    def embed_webpage(
        self,
        url: str,
        config: TextEmbedConfig | None = None,
        adapter: Adapter | None = None,
    ) -> list[EmbedData]:
        """
        Embeds the given webpage and returns a list of EmbedData objects.

        Args:
            url: The URL of the webpage to embed.
            config: The configuration for the embedding.
            adapter: The adapter for the embedding.

        Returns:
            A list of EmbedData objects.
        """

    def embed_directory(
        self,
        directory: str,
        config: TextEmbedConfig | None = None,
        adapter: Adapter | None = None,
    ) -> list[EmbedData]:
        """
        Embeds the given directory and returns a list of EmbedData objects.

        Args:
            directory: The path to the directory to embed.
            config: The configuration for the embedding.
            adapter: The adapter for the embedding.

        Returns:
            A list of EmbedData objects.
        """

    def embed_directory_stream(
        self,
        directory: str,
        config: TextEmbedConfig | None = None,
        adapter: Adapter | None = None,
    ) -> list[EmbedData]:
        """
        Embeds the given directory and returns a list of EmbedData objects.

        Args:
            directory: The path to the directory to embed.
            config: The configuration for the embedding.
            adapter: The adapter for the embedding.

        Returns:
            A list of EmbedData objects.
        """

    def embed_webpage(
        self,
        url: str,
        config: TextEmbedConfig | None = None,
        adapter: Adapter | None = None,
    ) -> list[EmbedData]:
        """
        Embeds the given webpage and returns a list of EmbedData objects.

        Args:
            url: The URL of the webpage to embed.
            config: The configuration for the embedding.
            adapter: The adapter for the embedding.

        Returns:
            A list of EmbedData objects.
        """

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
    CohereVision = ("CohereVision",)
    Bert = ("Bert",)
    Model2Vec = ("Model2Vec",)
    Jina = ("Jina",)
    Clip = ("Clip",)
    Colpali = ("Colpali",)
    ColBert = ("ColBert",)
    SparseBert = ("SparseBert",)
    ModernBert = ("ModernBert",)
    Qwen3 = ("Qwen3",)

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
    | `ModernBERTBase`                 | nomic-ai/modernbert-embed-base                   |
    | `ModernBERTLarge`                | nomic-ai/modernbert-embed-large                  |
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
    | `ModernBERTBase`                 | nomic-ai/modernbert-embed-base                   |
    ```
    """

    AllMiniLML6V2 = "AllMiniLML6V2"

    AllMiniLML6V2Q = "AllMiniLML6V2Q"

    AllMiniLML12V2 = "AllMiniLML12V2"

    AllMiniLML12V2Q = "AllMiniLML12V2Q"

    ModernBERTBase = "ModernBERTBase"

    ModernBERTLarge = "ModernBERTLarge"

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

    ModernBERTBase = "ModernBERTBase"
