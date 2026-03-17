from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest

from src.api.loaders.pdf_loader import PdfLoader


def _make_page(page_number: int, text: str | None) -> MagicMock:
    page = MagicMock()
    page.page_number = page_number
    page.extract_text.return_value = text
    return page


def _make_pdf_context(*pages: MagicMock) -> MagicMock:
    """Return a context-manager mock whose .pages is the given list."""
    pdf = MagicMock()
    pdf.pages = list(pages)
    cm = MagicMock()
    cm.__enter__ = MagicMock(return_value=pdf)
    cm.__exit__ = MagicMock(return_value=False)
    return cm


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_load_single_page():
    cm = _make_pdf_context(_make_page(1, "Hello world"))

    with patch("src.api.loaders.pdf_loader.pdfplumber.open", return_value=cm):
        results = PdfLoader().load("dummy.pdf")

    assert len(results) == 1
    text, metadata = results[0]
    assert text == "Hello world"
    assert metadata == {"page_number": 1, "source_type": "pdf"}


def test_load_multiple_pages():
    cm = _make_pdf_context(
        _make_page(1, "Page one text"),
        _make_page(2, "Page two text"),
        _make_page(3, "Page three text"),
    )

    with patch("src.api.loaders.pdf_loader.pdfplumber.open", return_value=cm):
        results = PdfLoader().load("dummy.pdf")

    assert len(results) == 3
    assert results[1][0] == "Page two text"
    assert results[1][1]["page_number"] == 2


def test_skips_pages_with_no_text():
    cm = _make_pdf_context(
        _make_page(1, "Real content"),
        _make_page(2, None),          # no extractable text
        _make_page(3, "   "),         # whitespace only
        _make_page(4, "More content"),
    )

    with patch("src.api.loaders.pdf_loader.pdfplumber.open", return_value=cm):
        results = PdfLoader().load("dummy.pdf")

    assert len(results) == 2
    assert results[0][1]["page_number"] == 1
    assert results[1][1]["page_number"] == 4


def test_empty_pdf_returns_empty_list():
    cm = _make_pdf_context()  # no pages

    with patch("src.api.loaders.pdf_loader.pdfplumber.open", return_value=cm):
        results = PdfLoader().load("empty.pdf")

    assert results == []


def test_text_is_stripped():
    cm = _make_pdf_context(_make_page(1, "  trimmed  "))

    with patch("src.api.loaders.pdf_loader.pdfplumber.open", return_value=cm):
        results = PdfLoader().load("dummy.pdf")

    assert results[0][0] == "trimmed"


def test_metadata_source_type_is_pdf():
    cm = _make_pdf_context(_make_page(1, "text"))

    with patch("src.api.loaders.pdf_loader.pdfplumber.open", return_value=cm):
        results = PdfLoader().load("dummy.pdf")

    assert results[0][1]["source_type"] == "pdf"
