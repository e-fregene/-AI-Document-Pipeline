import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from llama_index.llms.groq import Groq
import pandas as pd

load_dotenv()

llm       = Groq(model="qwen/qwen3.8-27b", api_key=os.getenv("GROQ_API_KEY"), max_tokens=30)
llm_batch = Groq(model="qwen/qwen3.8-27b", api_key=os.getenv("GROQ_API_KEY"), max_tokens=400)

BATCH_SIZE = 10  # pages per batch classification call


def classify_page(text: str) -> str:
    """Single-page classification — used as fallback when batch parse fails."""
    if not text or len(text.strip()) < 8:
        return "blank"
    prompt = f"""Classify the document page below into one of these types:
cover | toc | financial_table | general_text | chart | appendix | resume | contract | lender_fee_sheet | blank

Format: one label, all lowercase, words joined by underscores.
Example: lender_fee_sheet

Fallback if Finance related: if the page does not match any label above, invent the closest short descriptive label using the same format (e.g. payslip, tax_form).
Non Financial Fallback: If a page doesn't seem to fit in one of these categories and is not financial related, give it a title that best fits, stick to general classifiers or the closest one that already exists.

Page content:
{text[:600]}

Respond with only the label. No explanation."""
    return llm.complete(prompt).text.strip().lower()


def batch_classify_pages(pages: list[dict]) -> list[str]:
    """Classify all pages in one LLM call. Falls back per-page if response can't be parsed."""
    page_blocks = []
    for i, page in enumerate(pages):
        text = page["text"]
        if not text or len(text.strip()) < 8:
            page_blocks.append(f"--- Page {i + 1} ---\n[blank page]")
        else:
            page_blocks.append(f"--- Page {i + 1} ---\n{text[:400]}")

    prompt = f"""/no_think
Classify each document page below. Return exactly one label per page, one per line, in order.

Types: cover | toc | financial_table | general_text | chart | appendix | resume | contract | lender_fee_sheet | blank
Format: one label per line, all lowercase, words joined by underscores.
Fallback: if no type fits, invent a short descriptive label in the same format (e.g. payslip, lecture_slide).

{chr(10).join(page_blocks)}

Labels ({len(pages)} total, one per line):"""

    try:
        response  = llm_batch.complete(prompt).text.strip()
        labels    = [line.strip().lower() for line in response.split("\n") if line.strip()]
        if len(labels) >= len(pages):
            return labels[:len(pages)]
        # partial — fill remaining with single-page fallback
        return labels + [classify_page(pages[i]["text"]) for i in range(len(labels), len(pages))]
    except Exception:
        return [classify_page(page["text"]) for page in pages]


def is_same_document(prev_text: str, curr_text: str, doc_type: str = None) -> bool:
    """Ask the LLM whether two consecutive pages belong to the same logical document."""
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
    """Extract text then batch-classify all pages in chunks of BATCH_SIZE."""
    reader      = PdfReader(pdf_path)
    source_file = os.path.basename(pdf_path)

    raw_pages = []
    for i, page in enumerate(reader.pages):
        raw_pages.append({
            "page_num":    i,
            "source_file": source_file,
            "text":        page.extract_text() or "",
        })

    # Batch classify — ceil(N/BATCH_SIZE) LLM calls instead of N
    all_labels = []
    for start in range(0, len(raw_pages), BATCH_SIZE):
        batch  = raw_pages[start : start + BATCH_SIZE]
        labels = batch_classify_pages(batch)
        all_labels.extend(labels)

    for page, label in zip(raw_pages, all_labels):
        page["type"] = label

    return raw_pages


def route_documents(pdf_path: str) -> list[dict]:
    """Detect document boundaries and assign doc_id per logical document.
    Option 5: same-type consecutive pages skip the LLM boundary check entirely.
    Option 3: remaining boundary checks run in parallel via ThreadPoolExecutor."""
    doc_pages = load_pages(pdf_path)
    results   = []
    doc_counter = 0

    # Identify only the pairs where page type changes — only these need LLM boundary check
    type_change_indices = [
        i for i in range(1, len(doc_pages))
        if doc_pages[i]["type"] != doc_pages[i - 1]["type"]
    ]

    # Run those boundary checks in parallel
    boundary_results = {}
    if type_change_indices:
        def check_boundary(idx):
            prev = doc_pages[idx - 1]
            curr = doc_pages[idx]
            return idx, is_same_document(prev["text"], curr["text"], prev["type"])

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {executor.submit(check_boundary, idx): idx for idx in type_change_indices}
            for future in as_completed(futures):
                idx, same = future.result()
                boundary_results[idx] = same

    # Walk pages and assign doc_ids using precomputed boundary results
    current_page_type = None
    for i, page in enumerate(doc_pages):
        if i == 0:
            current_page_type = page["type"]
        else:
            same_type = page["type"] == doc_pages[i - 1]["type"]
            same_doc  = True if same_type else boundary_results.get(i, True)
            if not same_doc:
                doc_counter += 1
                current_page_type = page["type"]

        results.append({
            "page_num":    page["page_num"],
            "doc_id":      doc_counter,
            "page_type":   current_page_type,
            "source_file": page["source_file"],
            "text":        page["text"],
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
