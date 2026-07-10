import argparse
from typing import List, Optional

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

from search_r1.search.lance_retrieval import LanceRetriever


class Config:
    """Minimal config holder for LanceRetriever."""

    def __init__(
        self,
        lancedb_path: str = "tmp/lancedb",
        table_name: str = "docs",
        retrieval_topk: int = 10,
        corpus_path: Optional[str] = None,
    ):
        self.lancedb_path = lancedb_path
        self.table_name = table_name
        self.retrieval_topk = retrieval_topk
        self.corpus_path = corpus_path


class QueryRequest(BaseModel):
    queries: List[str]
    topk: Optional[int] = None
    return_scores: bool = False


app = FastAPI()
retriever: LanceRetriever
config: Config


@app.post("/retrieve")
def retrieve_endpoint(request: QueryRequest):
    """Batch retrieve documents for the provided queries."""
    topk = request.topk or config.retrieval_topk
    results, scores = retriever.batch_search(
        query_list=request.queries, num=topk, return_score=request.return_scores
    )

    formatted = []
    for i, docs in enumerate(results):
        if request.return_scores:
            formatted.append(
                [
                    {"document": doc, "score": score}
                    for doc, score in zip(docs, scores[i])
                ]
            )
        else:
            formatted.append(docs)
    return {"result": formatted}


def main():
    parser = argparse.ArgumentParser(description="LanceDB retriever server")
    parser.add_argument("--lancedb_path", type=str, default="tmp/lancedb")
    parser.add_argument("--table_name", type=str, default="docs")
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--corpus_path", type=str, default=None)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()

    global retriever, config
    config = Config(
        lancedb_path=args.lancedb_path,
        table_name=args.table_name,
        retrieval_topk=args.topk,
        corpus_path=args.corpus_path,
    )
    retriever = LanceRetriever(config)

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

