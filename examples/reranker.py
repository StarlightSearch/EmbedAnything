from embed_anything import JinaReranker, Dtype, RerankerResult

reranker = JinaReranker.from_pretrained("Xenova/bge-reranker-base", dtype=Dtype.FP16)

results = reranker.rerank(["What is the capital of France?"], ["France is a country in Europe.", "Paris is the capital of France."], 2)

print(results[0].documents)
