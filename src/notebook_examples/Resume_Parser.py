import pymupdf

def load_pdf(resume= "/Users/ethan/Desktop/-AI-Document-Pipeline/Ethan_Fregene_Resume (1) copy.pdf"):
    # Load the PDF
    doc = pymupdf.open(resume)

    # Print the number of pages
    print(f"Total Pages: {doc.page_count}")

    # Print metadata
    print("PDF Metadata:")

    doc.metadata


    """ Bounding Box Extractoion
    # Open the first page of the document
    page = doc[0]

    # Extract words along with bounding box information
    words = page.get_text("words")

    # Print first 5 extracted words with bounding boxes
    for word in words[:5]:
        print(word)
    """

    """ # Extract structured text as blocks)"""
    page = doc[0]
    blocks = page.get_text("blocks")

    # Print each block
    for block in blocks:
        print(f"Block: {block}\n")



if __name__ == "__main__":
    load_pdf()