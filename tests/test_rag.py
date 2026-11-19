"""Pipeline tests with a fake retriever and fake LLM: no Qdrant, model or API key needed."""
import asyncio

import pytest

from app import llm, rag, settings
from app.prompts import NO_ANSWER
from app.retrieval import Hit


def _hit(score, article="3", doc="399_2026_ND-CP_731671", title="Mức lương tối thiểu"):
    return Hit(score=score, payload={
        "doc_name": doc, "chapter": "Chương I", "article": article,
        "article_title": title, "clause": "", "text": f"Điều {article}. {title}\nNội dung.",
    })


@pytest.fixture(autouse=True)
def _setup(monkeypatch):
    monkeypatch.setattr(settings, "MIN_SCORE", 0.35)
    monkeypatch.setattr(settings, "LLM_PROVIDER", "mock")


def _patch(monkeypatch, hits, text="Mức lương là X [1]. Thêm ý [2]."):
    async def fake_retrieve(question, top_k, docs=None):
        return hits

    async def fake_generate(system, user):
        return text

    monkeypatch.setattr(rag, "retrieve_hits", fake_retrieve)
    monkeypatch.setattr(llm, "generate", fake_generate)


def test_extract_cited_ids_ignores_invalid_and_duplicates():
    assert rag.extract_cited_ids("a [2] b [1][2] c [9] d [0]", 3) == [2, 1]


def test_grounded_answer_returns_only_cited_sources(monkeypatch):
    _patch(monkeypatch, [_hit(0.8, "3"), _hit(0.7, "4"), _hit(0.6, "5")], "Đáp án [2].")
    res = asyncio.run(rag.answer("mức lương tối thiểu?"))
    assert res.grounded and [c.article for c in res.citations] == ["4"]
    assert res.confidence == 0.8 and res.refusal_reason is None
    assert res.citations[0].doc_label == "Nghị định 399/2026/NĐ-CP"


def test_refuses_below_threshold_without_calling_llm(monkeypatch):
    _patch(monkeypatch, [_hit(0.2)])

    async def boom(system, user):
        raise AssertionError("LLM must not be called")

    monkeypatch.setattr(llm, "generate", boom)
    res = asyncio.run(rag.answer("câu hỏi ngoài phạm vi"))
    assert res.answer == NO_ANSWER and not res.grounded
    assert res.refusal_reason == "low_retrieval_score" and res.citations == []


def test_model_declined_has_no_citations(monkeypatch):
    _patch(monkeypatch, [_hit(0.9)], NO_ANSWER)
    res = asyncio.run(rag.answer("câu hỏi"))
    assert not res.grounded and res.citations == [] and res.refusal_reason == "model_declined"


def test_answer_without_markers_is_flagged_ungrounded(monkeypatch):
    _patch(monkeypatch, [_hit(0.9)], "Trả lời nhưng quên trích dẫn.")
    res = asyncio.run(rag.answer("câu hỏi"))
    assert not res.grounded and res.refusal_reason == "no_citation"


def test_stream_event_order(monkeypatch):
    _patch(monkeypatch, [_hit(0.9)])

    async def fake_stream(system, user):
        for piece in ["Đáp ", "án [1]."]:
            yield piece

    monkeypatch.setattr(llm, "stream", fake_stream)

    async def collect():
        return [name async for name, _ in rag.answer_stream("câu hỏi")]

    names = asyncio.run(collect())
    assert names[0] == "sources" and names[-1] == "done" and names.count("token") == 2
