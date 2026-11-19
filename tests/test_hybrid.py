"""Retrieval tests: fusion, hybrid vs dense, reranking and its graceful fallback. All I/O is faked."""
from types import SimpleNamespace

import pytest

from app import clients, rag, retrieval, settings
from app.retrieval import Hit, rrf_fuse


def _pt(pid, score, article="1", text="văn bản"):
    return SimpleNamespace(id=pid, score=score, payload={"doc_name": "399_2026", "article": article, "text": text})


def test_rrf_rewards_agreement_between_rankings():
    fused = dict(rrf_fuse([["a", "b", "c"], ["c", "b", "d"]]))
    assert fused["b"] > fused["a"] and fused["b"] > fused["d"]
    assert list(fused)[0] in {"b", "c"}


@pytest.fixture
def faked(monkeypatch):
    monkeypatch.setattr(settings, "SEARCH_MODE", "hybrid")
    monkeypatch.setattr(settings, "RERANK", False)
    monkeypatch.setattr(retrieval, "_encode_query", lambda q: ([0.0], ([1, 2], [1.0, 1.0])))
    calls = []

    def fake_query(using, vector, limit, flt):
        calls.append((using, flt))
        if using == "dense":
            return [_pt("a", 0.6, "1"), _pt("b", 0.5, "2")]
        return [_pt("c", 3.0, "3"), _pt("b", 2.0, "2")]  # BM25: "c" only found by keywords

    monkeypatch.setattr(retrieval, "_query", fake_query)
    return calls


def test_hybrid_merges_both_searches_and_keeps_keyword_only_hits(faked):
    hits = retrieval.retrieve("mức lương tối thiểu", top_k=5)
    assert {h.payload["article"] for h in hits} == {"1", "2", "3"}
    assert hits[0].payload["article"] == "2"  # found by both retrievers
    assert [c[0] for c in faked] == ["dense", "bm25"]
    keyword_only = next(h for h in hits if h.payload["article"] == "3")
    assert keyword_only.score == 0.0 and keyword_only.dense is None


def test_dense_mode_skips_bm25(faked):
    retrieval.retrieve("mức lương", top_k=2, mode="dense")
    assert [c[0] for c in faked] == ["dense"]


def test_query_without_keywords_skips_bm25(faked, monkeypatch):
    monkeypatch.setattr(retrieval, "_encode_query", lambda q: ([0.0], ([], [])))
    retrieval.retrieve("???", top_k=2)
    assert [c[0] for c in faked] == ["dense"]


def test_rerank_reorders_and_replaces_score(faked, monkeypatch):
    monkeypatch.setattr(retrieval, "_rerank_scores", lambda q, texts: [0.1, 0.2, 0.9][: len(texts)])
    hits = retrieval.retrieve("mức lương", top_k=2, rerank=True)
    assert len(hits) == 2 and hits[0].rerank == 0.9 and hits[0].score == 0.9
    assert hits[0].rerank >= hits[1].rerank


def test_reranker_outage_degrades_to_unreranked_results(faked, monkeypatch):
    def down(q, texts):
        raise clients.ServiceError("reranker service unreachable")

    monkeypatch.setattr(retrieval, "_rerank_scores", down)
    hits = retrieval.retrieve("mức lương", top_k=3, rerank=True)
    assert len(hits) == 3 and all(h.rerank is None for h in hits)
    assert not rag._should_refuse([h for h in hits if h.dense and h.dense > 0.5])  # gate falls back to MIN_SCORE


def test_embedding_outage_is_a_retrieval_error(monkeypatch):
    def down(q):
        raise clients.ServiceError("embedding service unreachable")

    monkeypatch.setattr(retrieval, "_encode_query", down)
    with pytest.raises(retrieval.RetrievalError, match="embedding service"):
        retrieval.retrieve("q")


def test_docs_filter_is_forwarded(faked, monkeypatch):
    monkeypatch.setattr(retrieval, "_build_filter", lambda docs: ("FILTER", tuple(docs)) if docs else None)
    retrieval.retrieve("q", top_k=2, docs=["399_2026"])
    assert all(c[1] == ("FILTER", ("399_2026",)) for c in faked)


def test_gate_uses_best_score_not_first_hit():
    keyword_first = [Hit(score=0.0, payload={}), Hit(score=0.7, payload={}, dense=0.7)]
    assert not rag._should_refuse(keyword_first)
    assert rag._should_refuse([Hit(score=0.1, payload={}, dense=0.1)])


def test_reranked_hits_use_rerank_threshold(monkeypatch):
    monkeypatch.setattr(settings, "RERANK_MIN_SCORE", 0.05)
    monkeypatch.setattr(settings, "MIN_SCORE", 0.35)
    assert not rag._should_refuse([Hit(score=0.2, payload={}, rerank=0.2)])
    assert rag._should_refuse([Hit(score=0.01, payload={}, rerank=0.01)])
