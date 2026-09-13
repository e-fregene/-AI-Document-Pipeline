import pymupdf
import re
import json

PDF_PATH = "/Users/ethan/Desktop/-AI-Document-Pipeline/src/Data_Extract/LenderFeesWorksheetNew.pdf"

# ─────────────────────────────────────────────────────────────────────────────
# ARCHITECTURAL NOTE
# ─────────────────────────────────────────────────────────────────────────────
# This PDF is a form with two visual columns — labels on the left, values on the
# right. PyMuPDF's get_text("text") linearizes that into a single stream where
# all label text comes first, then all value text. That means we can't just look
# for "Interest Rate: 4.25%" on the same line, those two chunks are pages apart
# in the linearized output.
#
# Workflow chosen:
#   1. Header fields  → regex on the linearized text (values have unique patterns)
#   2. Fee rows       → regex on the same text (fee blocks have a predictable shape)
#   3. Bounding boxes → get_text("words") lets us pin any matched value to its
#                       position on the page after we've found it via text
#
# Why not find_tables()? PyMuPDF only detected 1 of the 2 fee sections as a table
# on this PDF. Regex on the raw text is more reliable here.
# ─────────────────────────────────────────────────────────────────────────────


def find_bbox(words, target_token):
    """
    Search the word list for a token and return its bounding box.
    Words from get_text("words"): (x0, y0, x1, y1, text, block_no, line_no, word_no)
    """
    for word in words:
        x0, y0, x1, y1, text, *_ = word
        if target_token.lower() in text.lower():
            return [round(x0), round(y0), round(x1), round(y1)]
    return None


def extract_header_fields(page):
    """
    Pull the six key scalar fields from the form header.

    The values sit in a predictable order after the 'FEES WORKSHEET' marker in
    the linearized text. We use regex patterns tuned to each field's format
    rather than line positions, so minor layout shifts don't break extraction.

    Fields identified for extraction:
      - Applicant Name  (identifies the customer)
      - Date Prepared   (document freshness)
      - Loan Program    (loan product type)
      - Total Loan Amount (core financial figure)
      - Interest Rate   (key comparison metric)
      - Loan Term       (repayment horizon)
    """
    text = page.get_text("text")
    words = page.get_text("words")

    # Isolate the value block — everything after the 'FEES WORKSHEET' header
    value_block = text[text.find("FEES WORKSHEET"):]

    # Each pattern targets the unique format of that field's value.
    # Anchoring to the value format (not the label) is more robust here because
    # the label and value aren't adjacent in the linearized stream.
    field_patterns = {
        "Applicant Name":    r"FEES WORKSHEET\n(.+)",
        "Date Prepared":     r"(\d{2}/\d{2}/\d{4})",
        "Loan Program":      r"(\d+ YEAR [A-Z\s\-]+Purchase)",
        "Total Loan Amount": r"\$\s+([\d,]+)\n",
        "Interest Rate":     r"(\d+\.\d+\s*%)",
        "Loan Term":         r"(\d+ / \d+ mths)",
    }

    results = []
    for label, pattern in field_patterns.items():
        match = re.search(pattern, value_block)
        if not match:
            results.append({"field": label, "text": "NOT FOUND", "bbox": None})
            continue

        value = match.group(1).strip()
        # Use the first token of the value to locate its bbox in the word list
        first_token = value.split()[0].replace(",", "").replace("$", "")
        bbox = find_bbox(words, first_token)

        results.append({"field": label, "text": value, "bbox": bbox})

    return results


def extract_fee_rows(page):
    """
    Extract individual fee line items from Origination Charges and Other Charges.

    Fee rows in the linearized text follow a consistent shape:
        Fee Name
        Paid To (lender or agent)
        Borrower
        $
        Amount

    Regex captures fee name + amount in one pass. The 'Borrower' anchor is what
    makes this reliable — it's always present in the Paid By column for these rows.
    """
    text = page.get_text("text")
    words = page.get_text("words")

    # Track which section we're in to label each fee correctly
    results = []
    current_section = None

    for section_name in ["ORIGINATION CHARGES", "OTHER CHARGES"]:
        section_start = text.find(section_name)
        if section_start == -1:
            continue

        # Find the next section boundary (or end of text)
        next_section = len(text)
        for other in ["ORIGINATION CHARGES", "OTHER CHARGES"]:
            pos = text.find(other, section_start + 1)
            if pos != -1 and pos < next_section:
                next_section = pos

        section_text = text[section_start:next_section]

        # Each fee row: fee name on one line, then paid-to, then "Borrower", then "$", then amount
        pattern = r"([A-Z][^\n$]+)\n[^\n]+\nBorrower\n\$\n([\d,.]+)"
        for match in re.finditer(pattern, section_text):
            fee_name = match.group(1).strip()
            amount = match.group(2).strip()

            # Skip section header lines that might match
            if fee_name in ("ORIGINATION CHARGES", "OTHER CHARGES"):
                continue

            # Anchor bbox to first word of the fee name
            bbox = find_bbox(words, fee_name.split()[0])

            results.append({
                "text": f"{fee_name} - ${amount}",
                "bbox": bbox,
                "section": section_name
            })

    return results


def run_extraction(pdf_path=PDF_PATH):
    doc = pymupdf.open(pdf_path)
    page = doc[0]

    print("=== HEADER FIELDS ===")
    header = extract_header_fields(page)
    for f in header:
        print(f"  {f['field']}: {f['text']}  |  bbox: {f['bbox']}")

    print("\n=== FEE ROWS ===")
    fees = extract_fee_rows(page)
    for f in fees:
        print(f"  [{f['section']}]  {f['text']}  |  bbox: {f['bbox']}")

    # Combined output in the required format
    combined = []
    for f in header:
        combined.append({"text": f"{f['field']}: {f['text']}", "bbox": f["bbox"]})
    for f in fees:
        combined.append({"text": f["text"], "bbox": f["bbox"]})

    print("\n=== COMBINED JSON OUTPUT ===")
    print(json.dumps(combined, indent=2))

    doc.close()
    return combined


# ─────────────────────────────────────────────────────────────────────────────
# REFLECTION
# ─────────────────────────────────────────────────────────────────────────────
# What worked:
#   - Regex on linearized text was fast to write and reliable for both header
#     fields (unique formats like "4.250 %" or "360 / 360 mths") and fee rows
#     (the Borrower/$/Amount pattern is consistent across all rows)
#   - get_text("words") gave clean bounding boxes for any token we wanted to pin
#
# What was harder:
#   - The two-column layout means labels and values aren't adjacent in the text
#     stream — you have to know what the value looks like to find it, not just
#     look next to the label
#   - Application No ("samplesmith") has no distinctive format — in a real
#     pipeline you'd need the document spec or a labeled training example
#   - find_tables() only caught one of the two fee sections; regex was more
#     reliable on this specific PDF but may need tuning on differently-formatted
#     lender worksheets
#
# Key takeaway for the pipeline:
#   PyMuPDF handles the hard part (rendering + coordinate mapping). Your code
#   owns the semantic layer — deciding what "Interest Rate" looks like and where
#   to look for it. In a production pipeline, that semantic layer gets replaced
#   by an LLM or a trained field extractor, and PyMuPDF becomes just the
#   text/bbox provider feeding it.
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_extraction()
