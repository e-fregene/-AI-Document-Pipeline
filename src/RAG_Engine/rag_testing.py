import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.dirname(__file__))

from llama_index.core import VectorStoreIndex
from llama_index.core.schema import Document
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from rag_pipeline import (
    EMBED_MODEL_NAME, PDF_PATH,
    route_documents, group_by_doc_id, documents_from_routed,
    build_index, load_model, expand_query, hybrid_retrieve, rerank,
)


def compare_embedding_models(documents: list[Document]):
    """Build a retrieval index with each embedding model and compare top chunks per query."""
    embedding_models = {
        "MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
        "BGE-small-en":  "BAAI/bge-small-en-v1.5",
        "E5-small-v2":   "intfloat/e5-small-v2",
    }
    test_queries = [
        "What is the loan amount?",
        "What fees does the borrower pay?",
        "What is the interest rate?",
    ]

    for model_name, model_id in embedding_models.items():
        print(f"\n{'=' * 60}")
        print(f"MODEL: {model_name}  ({model_id})")
        print('=' * 60)

        embed_model = HuggingFaceEmbedding(model_name=model_id)
        splitter    = SemanticSplitterNodeParser(embed_model=embed_model)
        nodes       = splitter.get_nodes_from_documents(documents)
        index       = VectorStoreIndex(nodes, embed_model=embed_model)
        retriever   = VectorIndexRetriever(index=index, similarity_top_k=3)

        for q in test_queries:
            print(f"\n  Q: {q}")
            results = retriever.retrieve(q)
            for i, r in enumerate(results):
                print(f"    [{i+1}] score={r.score:.4f} | {r.node.text[:150].strip()}")


def run_experiments(index: VectorStoreIndex):
    """Test retrieval quality with the finalised hybrid + rerank configuration."""
    experiments = [
        {"name": "H — Threshold + Rerank", "top_k": 8, "threshold": 0.6, "rerank": True},
    ]
    test_queries = [
        "What is the estimated total cost of CFPB regulations to consumers?",
        "How does CFPB oversight affect credit availability or loan access?",
        "What methodology was used to estimate the costs?",
    ]
    embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)
    llm = load_model()

    for exp in experiments:
        print(f"\n{'=' * 60}")
        print(f"EXPERIMENT {exp['name']}  |  top_k={exp['top_k']}  threshold={exp['threshold']}  reranker={exp['rerank']}")
        print('=' * 60)

        for q in test_queries:
            expanded  = expand_query(q, llm)
            nodes     = hybrid_retrieve(index, expanded, embed_model)
            top_nodes = rerank(nodes, q)

            print(f"\n  Q: {q}  →  {len(top_nodes)} chunk(s) returned")
            for i, n in enumerate(top_nodes):
                score_str = f"{n.score:.4f}" if n.score is not None else "  N/A "
                print(f"    [{i+1}] score={score_str} | {n.node.text[:120].strip()}")

        print()


if __name__ == "__main__":
    routed  = route_documents(PDF_PATH)
    grouped = group_by_doc_id(routed)
    docs    = documents_from_routed(grouped)
    index   = build_index(docs)
    run_experiments(index)
