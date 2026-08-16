"""
Tests for Data Schemas: CorpusEntry, QueryEntry, DatasetSplit.
"""
import pytest
from src.data.schemas import CorpusEntry, DatasetSplit, QueryEntry


def test_corpus_entry_properties_and_serialization():
    # Test title + text concatenation
    entry = CorpusEntry(doc_id="d1", title="Paper Title", text="Paper Abstract text.")
    assert entry.doc_id == "d1"
    assert entry.id == "d1"
    assert entry.full_text == "Paper Title Paper Abstract text."

    # Test serialization
    data = entry.to_dict()
    assert data["_id"] == "d1"
    assert data["title"] == "Paper Title"
    assert data["text"] == "Paper Abstract text."

    # Test deserialization
    restored = CorpusEntry.from_dict(data)
    assert restored.doc_id == "d1"
    assert restored.full_text == "Paper Title Paper Abstract text."

    # Test title-less entry
    entry_no_title = CorpusEntry(doc_id="d2", text="Only body text.")
    assert entry_no_title.full_text == "Only body text."


def test_query_entry_properties_and_serialization():
    q = QueryEntry(query_id="q1", text="Search query text")
    assert q.query_id == "q1"
    assert q.id == "q1"

    data = q.to_dict()
    assert data["_id"] == "q1"
    assert data["text"] == "Search query text"

    restored = QueryEntry.from_dict(data)
    assert restored.query_id == "q1"
    assert restored.text == "Search query text"


def test_dataset_split_accessors():
    corpus = {
        "d1": CorpusEntry(doc_id="d1", title="T1", text="Text 1"),
        "d2": CorpusEntry(doc_id="d2", title="", text="Text 2"),
    }
    queries = {
        "q1": QueryEntry(query_id="q1", text="Query 1"),
        "q2": QueryEntry(query_id="q2", text="Query 2"),
    }
    qrels = {
        "q1": {"d1": 1},
        "q2": {},  # unjudged query
    }
    split = DatasetSplit(name="mini", corpus=corpus, queries=queries, qrels=qrels)

    assert split.num_docs == 2
    assert split.num_queries == 2
    assert split.num_judged_queries == 1
    assert split.num_judgments == 1

    doc_ids, texts = split.get_corpus_texts()
    assert doc_ids == ["d1", "d2"]
    assert texts == ["T1 Text 1", "Text 2"]

    # get_query_texts only_judged=True
    judged_qids, judged_texts = split.get_query_texts(only_judged=True)
    assert judged_qids == ["q1"]
    assert judged_texts == ["Query 1"]

    # get_query_texts only_judged=False
    all_qids, all_texts = split.get_query_texts(only_judged=False)
    assert all_qids == ["q1", "q2"]
    assert all_texts == ["Query 1", "Query 2"]
