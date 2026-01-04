"""
Pdf managment tools.
"""

import arxiv
from typing import Optional, cast
import fitz  # PyMuPDF
import requests
from langchain.tools import tool

def search_pdf(query: str) -> Optional[str]:
    """
    Search arXiv for a paper and return the direct PDF URL.

    Use this when the user asks about a specific paper.
    """
    try:
        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=1,
            sort_by=arxiv.SortCriterion.Relevance
        )

        results = list(client.results(search))
        if not results:
            return None

        paper = results[0]
        return paper.pdf_url

    except Exception as e:
        return None

def download_pdf(pdf_url: str) -> Optional[bytes]:
    """
    Download a PDF from a URL and return raw bytes.
    """
    try:
        headers: dict[str, str] = {
            "User-Agent": "Mozilla/5.0"
        }
        response = requests.get(pdf_url, headers=headers, timeout=30)
        response.raise_for_status()

        if "application/pdf" not in response.headers.get("Content-Type", "").lower():
            return None

        return response.content

    except requests.exceptions.RequestException:
        return None

def extract_text_from_pdf(pdf_bytes: bytes) -> str: 
    """
    Extract all text from a PDF provided as raw bytes.
    """
    try:
        text_content: list[str] = []
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            for page in doc:
                page_text = cast(str, page.get_text("text"))
                text_content.append(page_text)
        return "\n".join(text_content)

    except Exception:
        return ""


@tool
def get_arxiv_pdf_content(query: str) -> Optional[str]:
    """
    Search arXiv for a paper by query, download it, and extract its text content.
    Use this to answer questions about the contents of scientific papers.
    """
    pdf_url = search_pdf(query)
    if not pdf_url:
        return "Could not find a paper for that query."
    
    pdf_bytes = download_pdf(pdf_url)
    if not pdf_bytes:
        return "Could not download the PDF."
    
    pdf_content = extract_text_from_pdf(pdf_bytes)
    return pdf_content if pdf_content else "Could not extract text from the PDF."