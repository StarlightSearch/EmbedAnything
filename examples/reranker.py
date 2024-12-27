from embed_anything import JinaReranker, Dtype, RerankerResult, DocumentRank

reranker = JinaReranker.from_pretrained("jinaai/jina-reranker-v1-turbo-en", dtype=Dtype.FP16)

results: RerankerResult = reranker.rerank(["What is the capital of France?"], ["France is a country in Europe.", "Paris is the capital of France."], 2)

documents: list[DocumentRank] = results[0].documents

print(documents)
