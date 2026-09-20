from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.gemini import Gemini

load_dotenv()

PDF_PATH = "/Users/ethan/Desktop/-AI-Document-Pipeline/src/data/MTG_10009588.pdf"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
GEMINI_MODEL = "models/gemini-3.6-flash"


def load_documents(pdf_path= PDF_PATH):
    documents = SimpleDirectoryReader(input_files=[pdf_path]).load_data()
    print(f"Loaded {len(documents)} document(s)")
    return documents


def build_index(documents) -> VectorStoreIndex:
    """Semantic Chunk then store in vector index"""
    embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)
    splitter = SemanticSplitterNodeParser(embed_model=embed_model)
    nodes = splitter.get_nodes_from_documents(documents)
    print(f"Total Semantic chunks created: {len(nodes)}")
    print(f"Sample chunk:\n{nodes[0].text[:300]}\n")

    # SemanticSplitterNodeParser already uses the embed_model
    # VectorStoreIndex re-embeds the final nodes for the vector store
    index = VectorStoreIndex(nodes, embed_model=embed_model)
    return index


def query(index: VectorStoreIndex, question: str) -> str:
    """Retrieve relevant chunks then generate answer with Gemini."""
    llm = Gemini(model=GEMINI_MODEL)
    engine = index.as_query_engine(llm=llm, similarity_top_k=2)
    response = engine.query(question)

    print(f"Q: {question}")
    print(f"A: {response.response}")
    print("\nSource chunks used:")
    for i, node in enumerate(response.source_nodes):
        print(f"  [{i+1}] (score: {node.score:.4f}) {node.text[:200]}")
    return response.response


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
