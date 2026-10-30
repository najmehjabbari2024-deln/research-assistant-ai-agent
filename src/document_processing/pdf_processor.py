import pymupdf
import re
from pathlib import Path


def extract_text_from_pdf(pdf_path, source_name=None):
    """
    Extract text from a PDF while preserving page metadata.
    Exclude the References/Bibliography section.
    """

    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF file not found: {pdf_path}"
        )

    extracted_pages = []
    references_started = False

    with pymupdf.open(pdf_path) as document:

        for page_number, page in enumerate(document, start=1):

            page_text = page.get_text("text")

            if references_started:
                continue

            references_match = re.search(
                r"(?im)^\s*(references|bibliography)\s*$",
                page_text
            )

            if references_match:
                page_text = page_text[:references_match.start()]
                references_started = True

            if page_text.strip():

                extracted_pages.append({
                    "source": source_name or pdf_path.name,
                    "page": page_number,
                    "text": page_text.strip()
                })

    return extracted_pages