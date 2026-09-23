import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from dotenv import load_dotenv

from llama_index.core import VectorStoreIndex
from llama_index.core.schema import Document
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core import QueryBundle
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from llama_index.llms.groq import Groq

from llama_index.retrievers.bm25 import BM25Retriever
from Data_Extract.PyMuPDF_Extraction import PDFExtractor
from Data_Extract.OCR_comparisons import render_page, PaddleOCRExtractor

load_dotenv()

PDF_PATH = "/Users/ethan/Downloads/Estimating-the-Cost-of-the-Consumer-Financial-Protection-Bureau-to-Consumers.pdf"
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"  # BGE: better domain discrimination. score spread reveals true gaps
LLM_MODEL = "mixtral-8x7b-32768"




def load_documents(pdf_path: str = PDF_PATH, use_ocr: bool = False) -> list[Document]:
    """Extract text per page. use_ocr=True switches to PaddleOCR for image-based PDFs."""
    import pymupdf
    doc = pymupdf.open(pdf_path)
    ocr = PaddleOCRExtractor() if use_ocr else None
    documents = []
    for page_num in range(len(doc)):
        if use_ocr:
            img = render_page(pdf_path, page_num)
            spans = ocr.extract(img)
        else:
            extractor = PDFExtractor(pdf_path)
            spans = extractor.extract_text_with_bbox(page_num)
            extractor.close()
        page_text = " ".join(s["text"] for s in spans)
        if page_text.strip():
            documents.append(Document(
                text=page_text,
                metadata={"page": page_num, "source": pdf_path}
            ))
    doc.close()
    print(f"Loaded {len(documents)} page(s) from {os.path.basename(pdf_path)}")
    return documents


def build_index(documents: list[Document]) -> VectorStoreIndex:
    """Semantic-chunk the documents then store embeddings in an in-memory vector index."""
    embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)
    splitter = SemanticSplitterNodeParser(embed_model=embed_model)
    nodes = splitter.get_nodes_from_documents(documents)
    if not nodes:
        raise ValueError("No chunks created — PDF may be image-based with no extractable text.")
    print(f"Total semantic chunks: {len(nodes)}")
    print(f"Sample chunk:\n{nodes[0].text[:300]}\n")
    return VectorStoreIndex(nodes, embed_model=embed_model)


def load_model(model: str = LLM_MODEL) -> Groq:
    """Mixtral-8x7B via Groq API — cloud-hosted Mistral, free tier, no local hardware needed."""
    return Groq(model=model, api_key=os.getenv("GROQ_API_KEY"))


def expand_query(question: str, llm: Ollama) -> list[str]:
    """Ask Gemini to rewrite the question 2 ways to improve retrieval coverage."""
    prompt = (
        "Rewrite this question in 2 different ways to improve document retrieval. "
        "Return only the rewritten questions, one per line:\n\n" + question
    )
    result = llm.complete(prompt)
    variants = [q.strip() for q in result.text.strip().split("\n") if q.strip()]
    return [question] + variants[:2]


def hybrid_retrieve(index: VectorStoreIndex, queries: list[str],
                    embed_model: HuggingFaceEmbedding, top_k: int = 5, threshold: float = 0.6):
    """Vector + BM25(keyword) retrieval across all expanded queries, deduplicated by node ID.
    Threshold only applied to vector results — BM25 keyword matches always included."""
    vector_retriever = VectorIndexRetriever(index=index, similarity_top_k=top_k)
    bm25_retriever = BM25Retriever.from_defaults(
        nodes=list(index.docstore.docs.values()), similarity_top_k=top_k
    )
    seen, nodes = set(), []
    for q in queries:
        for node in vector_retriever.retrieve(q):
            if node.score >= threshold and node.node_id not in seen:
                seen.add(node.node_id)
                nodes.append(node)
        for node in bm25_retriever.retrieve(q):
            if node.node_id not in seen:
                seen.add(node.node_id)
                nodes.append(node)
    return nodes


def rerank(nodes, question: str, top_n: int = 3):
    """Cross-encoder re-scores retrieved chunks by actual relevance to the question."""
    reranker = SentenceTransformerRerank(
        model="cross-encoder/ms-marco-MiniLM-L-2-v2", top_n=top_n
    )
    return reranker.postprocess_nodes(nodes, query_bundle=QueryBundle(question))


def query(index: VectorStoreIndex, question: str) -> str:
    """Full RAG pipeline: expand query → hybrid retrieve → rerank → Mistral answer."""
    llm = load_model()
    embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)

    expanded = expand_query(question, llm)
    nodes = hybrid_retrieve(index, expanded, embed_model)
    top_nodes = rerank(nodes, question)

    context = "\n\n".join(n.node.text for n in top_nodes)
    prompt = f"Answer based on the document context below:\n\n{context}\n\nQuestion: {question}"
    answer = llm.complete(prompt).text.strip()

    print(f"\nQ: {question}")
    print(f"A: {answer}")
    print("\nSource chunks used:")
    for i, n in enumerate(top_nodes):
        print(f"  [{i+1}] (score: {n.score:.4f}) {n.node.text[:200]}")
    return answer


def compare_embedding_models(documents: list[Document]):
    """Build a retrieval index with each embedding model and compare top chunks per query."""
    embedding_models = {
        "MiniLM-L6-v2":  "sentence-transformers/all-MiniLM-L6-v2",
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
        splitter = SemanticSplitterNodeParser(embed_model=embed_model)
        nodes = splitter.get_nodes_from_documents(documents)
        index = VectorStoreIndex(nodes, embed_model=embed_model)
        retriever = VectorIndexRetriever(index=index, similarity_top_k=3)

        for q in test_queries:
            print(f"\n  Q: {q}")
            results = retriever.retrieve(q)
            for i, r in enumerate(results):
                print(f"    [{i+1}] score={r.score:.4f} | {r.node.text[:150].strip()}")


def run_experiments(index: VectorStoreIndex):
    """Test retrieval quality across top_k, score threshold, and reranker combos."""
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
            expanded = expand_query(q, llm)
            nodes= hybrid_retrieve(index, expanded, embed_model)
            top_nodes = rerank(nodes, q)

            print(f"\n  Q: {q}  →  {len(top_nodes)} chunk(s) returned")
            for i, n in enumerate(top_nodes):
                score_str = f"{n.score:.4f}" if n.score is not None else "  N/A "
                print(f"    [{i+1}] score={score_str} | {n.node.text[:120].strip()}")

        print()


if __name__ == "__main__":
    docs = load_documents(use_ocr=False)
    index = build_index(docs)
    run_experiments(index)

