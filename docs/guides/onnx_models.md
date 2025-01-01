# Using ONNX Models

## Supported Models

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
| `JINAV3`                         | jinaai/jina-embeddings-v3                         |

## Example Usage

``` python
--8<-- "examples/onnx_models.py"
```
