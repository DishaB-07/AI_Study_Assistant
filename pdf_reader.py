import easyocr
import fitz
import numpy as np

reader = easyocr.Reader(['en'], gpu=False)

def extract_text_from_pdf(pdf_path):

    doc = fitz.open(pdf_path)
    text_chunks = []

    for page_number in range(len(doc)):

        page = doc.load_page(page_number)
        pix = page.get_pixmap()

        img = np.frombuffer(pix.samples, dtype=np.uint8)
        img = img.reshape(pix.height, pix.width, pix.n)

        result = reader.readtext(img)

        text = " ".join([item[1] for item in result])

        if text.strip():
            text_chunks.append((page_number + 1, text))

    return text_chunks