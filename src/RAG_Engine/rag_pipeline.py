import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from dotenv import load_dotenv

from llama_index.core import VectorStoreIndex
from llama_index.core.schema import Document
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core import QueryBundle
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from llama_index.llms.groq import Groq

from llama_index.retrievers.bm25 import BM25Retriever
from Data_Extract.PyMuPDF_Extraction import PDFExtractor
from Data_Extract.OCR_comparisons import render_page, PaddleOCRExtractor
from document_processor import route_documents

load_dotenv()

PDF_PATH = "src/data/Blob File Sample.pdf"
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"  # BGE: better domain discrimination. score spread reveals true gaps
LLM_MODEL = "qwen/qwen3.8-27b"




def group_by_doc_id(routed_pages: list[dict]) -> list[dict]:
    """Merge consecutive pages sharing the same doc_id into one combined text block."""
    groups = {}
    for page in routed_pages:
        doc_id = page["doc_id"]
        if doc_id not in groups:
            groups[doc_id] = {
                "doc_id":      doc_id,
                "page_type":   page["page_type"],
                "source_file": page["source_file"],
                "page_start":  page["page_num"],
                "page_end":    page["page_num"],
                "text":        page["text"],
            }
        else:
            groups[doc_id]["text"]     += "\n\n" + page["text"]
            groups[doc_id]["page_end"]  = page["page_num"]
    return list(groups.values())


def documents_from_routed(routed_pages: list[dict]) -> list[Document]:
    """Convert grouped or raw routed pages into LlamaIndex Documents with full metadata."""
    documents = []
    for page in routed_pages:
        if page["text"].strip():
            documents.append(Document(
                text=page["text"],
                metadata={
                    "page_start":  page.get("page_start", page.get("page_num")),
                    "page_end":    page.get("page_end",   page.get("page_num")),
                    "doc_type":    page["page_type"],
                    "source_file": page["source_file"],
                    "doc_id":      page["doc_id"],
                }
            ))
    return documents




def predict_doc_type(query: str, routed_pages: list[dict], llm) -> str:
    """Ask the LLM which doc_type in the blob is most relevant to the query."""
    seen = {}
    for page in routed_pages:
        doc_type = page["page_type"]
        if doc_type not in seen:
            seen[doc_type] = page["text"][:300]

    descriptions = "\n".join(
        f'- doc_type: "{doc_type}" | excerpt: "{excerpt}"'
        for doc_type, excerpt in seen.items()
    )
    prompt = f"""/no_think
User query: "{query}"

Available document types and excerpts:
{descriptions}

Which doc_type is most likely to contain the answer? Respond with only the doc_type label."""
    return llm.complete(prompt).text.strip().lower()


def retrieve_by_doc_type(routed_pages: list[dict], doc_type: str) -> list[dict]:
    """Return pages matching doc_type. Falls back to all pages if no match found."""
    matched = [page for page in routed_pages if page["page_type"] == doc_type]
    if not matched:
        print(f"No pages matched doc_type '{doc_type}' — falling back to full index search. Patience por favor")
        return routed_pages
    return matched


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
    """Qwen3-27B via Groq API — cloud-hosted, free tier, capped to stay within OTPM limits."""
    return Groq(model=model, api_key=os.getenv("GROQ_API_KEY"), max_tokens=500)


def expand_query(question: str, llm: LLM_MODEL) -> list[str]:
    """Ask Gemini to rewrite the question 2 ways to improve retrieval coverage."""
    prompt = (
        "Rewrite this question in 2 different ways to improve document retrieval. "
        "Return only the rewritten questions, one per line:\n\n" + question
    )
    result = llm.complete(prompt)
    variants = [q.strip() for q in result.text.strip().split("\n") if q.strip()]
    return [question] + variants[:2]


def hybrid_retrieve(index: VectorStoreIndex, queries: list[str],
                    embed_model: HuggingFaceEmbedding, top_k: int = 5,
                    threshold: float = 0.6, doc_type: str = None):
    """Vector + BM25 retrieval across all expanded queries, deduplicated by node ID.
    Vector retriever filters by doc_type at index level. BM25 filters post-retrieval."""
    if doc_type:
        filters = MetadataFilters(filters=[ExactMatchFilter(key="doc_type", value=doc_type)])
        vector_retriever = VectorIndexRetriever(index=index, similarity_top_k=top_k, filters=filters)
    else:
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
            if doc_type and node.node.metadata.get("doc_type") != doc_type:
                continue
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


def query(index: VectorStoreIndex, question: str, routed_pages: list[dict] = None) -> str:
    """Full RAG pipeline: predict doc_type → expand query → hybrid retrieve → rerank → answer."""
    llm = load_model()
    embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)

    predicted_type = predict_doc_type(question, routed_pages, llm) if routed_pages else None
    if predicted_type:
        print(f"Predicted doc_type: {predicted_type}")

    expanded  = expand_query(question, llm)
    nodes     = hybrid_retrieve(index, expanded, embed_model, doc_type=predicted_type)
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
    routed  = route_documents(PDF_PATH)
    grouped = group_by_doc_id(routed)
    docs    = documents_from_routed(grouped)
    index   = build_index(docs)

    test_questions = [
        "What does John smith work in",
        "What is Joe's Total pay",
        "What did John and mary purchase or borrow? a loan?",
    ]
    for question in test_questions:
        query(index, question, routed_pages=routed)

