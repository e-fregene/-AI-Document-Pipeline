import os
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from llama_index.llms.groq import Groq
import pandas as pd

pdf_path="/Users/ethan/Desktop/-AI-Document-Pipeline/src/data/Blob File Sample.pdf"
load_dotenv()
llm = Groq(model="qwen/qwen3.8-27b", api_key=os.getenv("GROQ_API_KEY"), max_tokens=30)


def classify_page(text: str) -> str:
    """Use the LLM to classify a page's type from its content."""
    if not text or len(text.strip()) < 8:
        return "blank"
    prompt = f"""Classify the document page below into one of these types:
cover | toc | financial_table | general_text | chart | appendix | resume | contract | lender_fee_sheet | blank

Format: one label, all lowercase, words joined by underscores.
Example: lender_fee_sheet

Fallback if Finance related: if the page does not match any label above, invent the closest short descriptive label using the same format (e.g. payslip, tax_form).
Non Fincial Fallback: If a page doesnt seem to fit in one of these categories and is not fincial related, give it a title that best fits, don't think to hard just stick to general classifers

Page content:
{text[:600]}

Respond with only the label.  No explanation."""
    return llm.complete(prompt).text.strip().lower()


def is_same_document(prev_text: str, curr_text: str, doc_type: str = None) -> bool:
    """Ask the LLM whether two consecutive pages belong to the same document."""
    prompt = f"""/no_think
Decide whether the two pages below belong to the same logical document.

Format: respond with only Yes or No.
Example: Yes

Previous page type: {doc_type or 'unknown'}

Previous page:
{prev_text[:500]}

Current page:
{curr_text[:500]}"""
    response = llm.complete(prompt).text.strip().lower()
    return response.startswith("yes")


def load_pages(pdf_path: str) -> list[dict]:
    """Extract text from each page and attach page number, source file, and type metadata."""
    reader = PdfReader(pdf_path)
    source_file = os.path.basename(pdf_path)
    doc_pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        doc_pages.append({
            "page_num": i,
            "source_file": source_file,
            "text": text,
            "type": classify_page(text),
        })
    return doc_pages


def route_documents(pdf_path: str) -> list[dict]:
    """Load pages then walk them, detecting document boundaries and assigning doc_id."""
    doc_pages = load_pages(pdf_path)
    results = []
    current_page_type = None
    doc_counter = 0

    for i, page in enumerate(doc_pages):
        if i == 0:
            current_page_type = page["type"]
        else:
            prev_text = doc_pages[i - 1]["text"]
            curr_text=page["text"]
            same = is_same_document(prev_text, curr_text, current_page_type)
            if not same:
                doc_counter += 1
                current_page_type = page["type"]

        results.append({
            "page_num": page["page_num"],
            "doc_id": doc_counter,
            "page_type": current_page_type,
            "source_file": page["source_file"],
            "text": page["text"],
        })

    
    df = pd.DataFrame(results)
    df.head()
    return results


def summarize(doc_pages: list[dict]):
    """Print a summary of page count and type distribution."""
    from collections import Counter
    counts = Counter(p.get("page_type") or p.get("type") for p in doc_pages)
    print(f"Total pages: {len(doc_pages)}")
    for page_type, count in counts.most_common():
        print(f"  {page_type}: {count}")


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else None
    if not path:
        print("Usage: python3 document_processor.py <path/to/file.pdf>")
        sys.exit(1)

    routed = route_documents(path)
    summarize(routed)
    print("\nRouted pages:")
    for r in routed:
        print(f"  [page {r['page_num']} | doc_id={r['doc_id']} | {r['page_type']}]")
