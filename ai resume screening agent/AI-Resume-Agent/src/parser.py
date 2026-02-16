from pathlib import Path
from typing import Optional
import logging

import PyPDF2
import docx

logger = logging.getLogger(__name__)


def extract_text_from_pdf(path: Path) -> str:
    """Extract text content from a PDF file.

    Args:
        path: Path to the PDF file.

    Returns:
        Extracted text as a single string.
    """
    try:
        text = []
        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text.append(page_text)
        return "\n".join(text)
    except Exception as e:
        logger.exception("Failed to extract PDF text: %s", e)
        return ""


def extract_text_from_docx(path: Path) -> str:
    """Extract text content from a DOCX file.

    Args:
        path: Path to the DOCX file.

    Returns:
        Extracted text as a single string.
    """
    try:
        doc = docx.Document(str(path))
        paragraphs = [p.text for p in doc.paragraphs if p.text]
        return "\n".join(paragraphs)
    except Exception as e:
        logger.exception("Failed to extract DOCX text: %s", e)
        return ""


def extract_text(path: str) -> str:
    """Generic text extractor for supported resume formats.

    Args:
        path: Path to resume file (PDF or DOCX).

    Returns:
        Extracted text.
    """
    p = Path(path)
    if not p.exists():
        logger.error("File not found: %s", path)
        return ""
    suffix = p.suffix.lower()
    if suffix == ".pdf":
        return extract_text_from_pdf(p)
    if suffix in (".docx", ".doc"):
        return extract_text_from_docx(p)
    logger.warning("Unsupported file type: %s", suffix)
    return ""
