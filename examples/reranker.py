from embed_anything import Reranker, Dtype, RerankerResult, DocumentRank

reranker = Reranker.from_pretrained("jinaai/jina-reranker-v1-turbo-en", dtype=Dtype.F16)

results: list[RerankerResult] = reranker.rerank(["What is the capital of France?"], ["France is a country in Europe.", "Paris is the capital of France."], 2)

for result in results:
    documents: list[DocumentRank] = result.documents
    print(documents)
