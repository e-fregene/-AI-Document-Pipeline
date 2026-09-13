import pymupdf  # PyMuPDF
import re
import cv2
import numpy as np
from PIL import Image

def extract_page(file_name= "/Users/ethan/Desktop/-AI-Document-Pipeline/src/data/LenderFeesWorksheetNew.pdf"):
    # Load the PDF
    doc = pymupdf.open(file_name)

    # Print metadata
    print("PDF Metadata:", doc.metadata)



    # Extract text from the first page
    page = doc[0]
    #print(page.get_text())



    #Extract text wwith bounding boxes
    text_data = page.get_text("words")  # Extract text as a structured dictionary

    for block in text_data.get("blocks", []):  # Loop through all blocks in the page
        bbox = block["bbox"]  # Get bounding box of the block
        
        # Check if 'lines' exists in the block
        if "lines" in block:
            for line in block["lines"]:
                for span in line.get("spans", []):  # Ensure spans exist
                    text = span["text"]
                    print(f"Text: {text}, Bounding Box: {bbox}")

    # Render the first page as an image
    pix = page.get_pixmap()
    pix.save("page1.png")


def extract_page_test(file_name= "/Users/ethan/Desktop/-AI-Document-Pipeline/src/data/LenderFeesWorksheetNew.pdf"):
    import json
    doc = pymupdf.open(file_name)
    results = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        text_data = page.get_text("dict")

        for block in text_data.get("blocks", []):
            if "lines" in block:
                for line in block["lines"]:
                    for span in line.get("spans", []):
                        if span["text"].strip():
                            results.append({
                                "text": span["text"].strip(),
                                "bbox": [round(v) for v in span["bbox"]]
                            })

    print(json.dumps(results, indent=2))

def regex_formatting_testing(text, file_name= "/Users/ethan/Desktop/-AI-Document-Pipeline/src/data/LenderFeesWorksheetNew.pdf"):
    text = """
    Name: John Smith
    Email: john.smi@example.com
    Phone: +1 (416) 555-1234
    LoanID: L-2025-0042
    Date of Birth: 2000-08-15
    Address: 456 Hudson St, Jersey City, NJ 07302
    """

    doc = pymupdf.open(file_name)
    text = doc[0].get_text("text")

    # Regex to find phone number
    phone_match = re.search(r"\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}", text)

    if phone_match:
        phone_number = phone_match.group()
        print(f"Candidate Phone Number: {phone_number}")
    else:
        print("Phone number not found.")

    words = doc[0].get_text("words")

    target_word = phone_number

    for word in words:
        x0, y0, x1, y1, text, block, line, word_no = word
        if target_word.lower() in text.lower():
            print(f"Found '{target_word}' at: ({x0}, {y0}, {x1}, {y1})")

    def draw_bounding_box():

    # Convert PDF page to an image
        pix = doc[0].get_pixmap()
        img = np.array(Image.frombytes("RGB", [pix.width, pix.height], pix.samples))

        # Convert image to OpenCV BGR format
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # Get the actual image height
        img_height = img.shape[0]  # OpenCV uses (height, width, channels)

        # Extract words and their bounding boxes
        words = doc[0].get_text("words")

        # Define the field we are looking for
        target_word = "123-456-7890"  # Replace with actual phone number

        # Flag to check if the word was found
        word_found = False

        # Search for the target word and retrieve its bounding box
        for word in words:
            x0, y0, x1, y1, text, block, line, word_no = word  # Unpack correctly

            if target_word in text:  # Case-sensitive match (modify if needed)
            # Convert PyMuPDF's y-coordinates (bottom-left origin) to OpenCV's (top-left origin)
                y0_new = y1  # Convert bottom-left to top-left
                y1_new = y0  # Convert bottom-left to top-left

        # Convert coordinates to integers
        x0, y0_new, x1, y1_new = map(int, [x0, y0_new, x1, y1_new])

        # Draw a rectangle around the detected word
        cv2.rectangle(img, (x0, y0_new), (x1, y1_new), (0, 255, 0), 2)

        print(f"Found '{target_word}' at: ({x0}, {y0_new}, {x1}, {y1_new})")
        word_found = True

        # Ensure an image is displayed even if no word is found
        if not word_found:
            print(f"'{target_word}' not found in document.")

        # Convert back to RGB for displaying in PIL
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Display the image using PIL (works in Jupyter/Colab)
        display(Image.fromarray(img_rgb))


def extract_tables_with_bbox(pdf_path):
    doc = pymupdf.open(pdf_path)
    results = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        tables = page.find_tables()

        for i, table in enumerate(tables.tables):
            table_bbox = table.bbox
            table_data = []
            cell_bboxes = []

            for row_num in range(table.row_count):
                row_data = []
                row_bboxes = []

                for col_num in range(table.col_count):
                    cell_idx = row_num * table.col_count + col_num

                    if cell_idx < len(table.cells) and table.cells[cell_idx] is not None:
                        cell_bbox = table.cells[cell_idx]
                        cell_text = page.get_text("text", clip=cell_bbox).strip()
                        row_data.append(cell_text)
                        row_bboxes.append(cell_bbox)
                    else:
                        row_data.append("")
                        row_bboxes.append(None)

                table_data.append(row_data)
                cell_bboxes.append(row_bboxes)

            results.append({
                'page': page_num,
                'table_index': i,
                'bbox': table_bbox,
                'data': table_data,
                'cell_bboxes': cell_bboxes
            })

    doc.close()

    for table in results:
        for row in table['data']:
            print(row)
    return results



if __name__ == "__main__":
    extract_page_test("/Users/ethan/Desktop/-AI-Document-Pipeline/src/data/LenderFeesWorksheetNew.pdf")