from unittest.mock import MagicMock, patch

from src.api.loaders.docx_loader import DocxLoader, PARAGRAPHS_PER_BLOCK


def _make_doc(texts: list[str]) -> MagicMock:
    """Return a mock Document whose .paragraphs matches the given text list."""
    paragraphs = []
    for t in texts:
        p = MagicMock()
        p.text = t
        paragraphs.append(p)
    doc = MagicMock()
    doc.paragraphs = paragraphs
    return doc


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_load_fewer_than_block_size():
    doc = _make_doc(["Para one", "Para two", "Para three"])

    with patch("src.api.loaders.docx_loader.Document", return_value=doc):
        results = DocxLoader().load("dummy.docx")

    assert len(results) == 1
    text, metadata = results[0]
    assert text == "Para one\nPara two\nPara three"
    assert metadata == {"paragraph_index": 0, "source_type": "docx"}


def test_load_exact_block_size():
    texts = [f"Paragraph {i}" for i in range(PARAGRAPHS_PER_BLOCK)]
    doc = _make_doc(texts)

    with patch("src.api.loaders.docx_loader.Document", return_value=doc):
        results = DocxLoader().load("dummy.docx")

    assert len(results) == 1
    assert results[0][1]["paragraph_index"] == 0


def test_load_multiple_blocks():
    texts = [f"Paragraph {i}" for i in range(PARAGRAPHS_PER_BLOCK * 2 + 1)]
    doc = _make_doc(texts)

    with patch("src.api.loaders.docx_loader.Document", return_value=doc):
        results = DocxLoader().load("dummy.docx")

    assert len(results) == 3
    assert results[0][1]["paragraph_index"] == 0
    assert results[1][1]["paragraph_index"] == PARAGRAPHS_PER_BLOCK
    assert results[2][1]["paragraph_index"] == PARAGRAPHS_PER_BLOCK * 2


def test_empty_paragraphs_are_skipped():
    doc = _make_doc(["Real text", "", "   ", "More text"])

    with patch("src.api.loaders.docx_loader.Document", return_value=doc):
        results = DocxLoader().load("dummy.docx")

    assert len(results) == 1
    assert "Real text" in results[0][0]
    assert "More text" in results[0][0]


def test_all_empty_paragraphs_returns_empty_list():
    doc = _make_doc(["", "   ", "\t"])

    with patch("src.api.loaders.docx_loader.Document", return_value=doc):
        results = DocxLoader().load("dummy.docx")

    assert results == []


def test_empty_document_returns_empty_list():
    doc = _make_doc([])

    with patch("src.api.loaders.docx_loader.Document", return_value=doc):
        results = DocxLoader().load("dummy.docx")

    assert results == []


def test_metadata_source_type_is_docx():
    doc = _make_doc(["Some text"])

    with patch("src.api.loaders.docx_loader.Document", return_value=doc):
        results = DocxLoader().load("dummy.docx")

    assert results[0][1]["source_type"] == "docx"


def test_block_text_joins_paragraphs_with_newline():
    doc = _make_doc(["First", "Second", "Third"])

    with patch("src.api.loaders.docx_loader.Document", return_value=doc):
        results = DocxLoader().load("dummy.docx")

    assert results[0][0] == "First\nSecond\nThird"
