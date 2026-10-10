from __future__ import annotations

from unittest.mock import MagicMock

from src.services.file_search_store import (
    _delete_stale_documents_for_source,
    _is_doc_for_source,
)


class DummyDoc:
    def __init__(self, name: str, display_name: str | None = None, custom_metadata: list | None = None):
        self.name = name
        self.display_name = display_name
        self.custom_metadata = custom_metadata or []


def test_is_doc_for_source_exact_display_name() -> None:
    doc = DummyDoc(name="doc1", display_name="report.pdf")
    assert _is_doc_for_source(doc, base_stem="report", source_filename="report.pdf") is True


def test_is_doc_for_source_slice_prefix() -> None:
    doc = DummyDoc(name="doc2", display_name="report__sec_01.pdf")
    assert _is_doc_for_source(doc, base_stem="report", source_filename="report.pdf") is True


def test_is_doc_for_source_custom_metadata() -> None:
    doc = DummyDoc(
        name="doc3",
        display_name="other_name.pdf",
        custom_metadata=[{"key": "source_filename", "string_value": "report.pdf"}],
    )
    assert _is_doc_for_source(doc, base_stem="report", source_filename="report.pdf") is True


def test_is_doc_for_source_unrelated() -> None:
    doc = DummyDoc(name="doc4", display_name="other_doc.pdf")
    assert _is_doc_for_source(doc, base_stem="report", source_filename="report.pdf") is False


def test_delete_stale_documents_for_source() -> None:
    mock_client = MagicMock()
    doc_match1 = DummyDoc(name="stores/s1/documents/d1", display_name="report__sec_01.pdf")
    doc_match2 = DummyDoc(name="stores/s1/documents/d2", display_name="report.pdf")
    doc_unrelated = DummyDoc(name="stores/s1/documents/d3", display_name="unrelated.pdf")

    mock_client.file_search_stores.documents.list.return_value = [
        doc_match1,
        doc_match2,
        doc_unrelated,
    ]

    deleted = _delete_stale_documents_for_source(
        client=mock_client,
        store_name="stores/s1",
        source_filename="report.pdf",
    )

    assert deleted == 2
    assert mock_client.file_search_stores.documents.delete.call_count == 2
    mock_client.file_search_stores.documents.delete.assert_any_call(name="stores/s1/documents/d1")
    mock_client.file_search_stores.documents.delete.assert_any_call(name="stores/s1/documents/d2")
