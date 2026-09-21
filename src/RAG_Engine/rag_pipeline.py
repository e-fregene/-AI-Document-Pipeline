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
from llama_index.llms.gemini import Gemini
from Data_Extract.PyMuPDF_Extraction import PDFExtractor

load_dotenv()

PDF_PATH = "/Users/ethan/Desktop/-AI-Document-Pipeline/src/data/LenderFeesWorksheetNew.pdf"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
GEMINI_MODEL = "models/gemini-3.6-flash"


def load_documents(pdf_path: str = PDF_PATH) -> list[Document]:
    """Extract text from each PDF page using PDFExtractor and wrap in Document objects."""
    extractor = PDFExtractor(pdf_path)
    metadata = extractor.get_metadata()
    documents = []
    for page_num in range(len(extractor.doc)):
        spans = extractor.extract_text_with_bbox(page_num)
        page_text = " ".join(s["text"] for s in spans)
        if page_text.strip():
            documents.append(Document(
                text=page_text,
                metadata={"page": page_num, "source": pdf_path, **metadata}
            ))
    extractor.close()
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


def expand_query(question: str, llm: Gemini) -> list[str]:
    """Ask Gemini to rewrite the question 2 ways to improve retrieval coverage."""
    prompt = (
        "Rewrite this question in 2 different ways to improve document retrieval. "
        "Return only the rewritten questions, one per line:\n\n" + question
    )
    result = llm.complete(prompt)
    variants = [q.strip() for q in result.text.strip().split("\n") if q.strip()]
    return [question] + variants[:2]


def hybrid_retrieve(index: VectorStoreIndex, queries: list[str],
                    embed_model: HuggingFaceEmbedding, top_k: int = 4):
    """Run all expanded queries through the vector retriever, deduplicate by node ID."""
    retriever = VectorIndexRetriever(index=index, similarity_top_k=top_k)
    seen, nodes = set(), []
    for q in queries:
        for node in retriever.retrieve(q):
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
    """Full RAG pipeline: expand query → hybrid retrieve → rerank → Gemini answer."""
    llm = Gemini(model=GEMINI_MODEL)
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


if __name__ == "__main__":
    docs = load_documents()
    index = build_index(docs)

    questions = [
        "What is this document about?",
        "Who are the borrowers?",
        "What is the loan amount?",
    ]
    for q in questions:
        query(index, q)
        print("\n" + "=" * 60 + "\n")

