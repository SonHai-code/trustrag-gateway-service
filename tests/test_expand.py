"""Small-to-big: a hit on one chunk of a long Điều must reach the LLM as the WHOLE Điều."""
from types import SimpleNamespace

import pytest

from app import rag, retrieval, settings
from app.expand import expandable, join_article
from app.retrieval import Hit

HEADER = "Điều 7. Quy định dài"


def _part(i, body, parts=4, overlap=0, article="7", doc="399_2026_ND-CP_1", clause=""):
    return {"doc_name": doc, "article": article, "article_title": "Quy định dài", "clause": clause or str(i + 1),
            "part": i, "parts": parts, "overlap": overlap, "text": f"{HEADER}\n{body}", "context": f"ctx > {HEADER}"}


FOUR = [_part(0, "1. một"), _part(1, "2. hai"), _part(2, "3. ba"), _part(3, "4. bốn")]


def test_join_puts_parts_in_order_and_keeps_one_header():
    text, truncated = join_article(list(reversed(FOUR)))
    assert text == f"{HEADER}\n1. một\n2. hai\n3. ba\n4. bốn" and not truncated


def test_join_removes_overlap_without_a_newline():
    a = _part(0, "Câu một dài. Câu hai dài.", 2)
    b = _part(1, "Câu hai dài. Câu ba dài.", 2, overlap=len("Câu hai dài."))
    assert join_article([a, b])[0] == f"{HEADER}\nCâu một dài. Câu hai dài. Câu ba dài."


def test_join_keeps_content_that_merely_looks_like_a_header():
    p = _part(1, "x", 2)
    p["text"] = "Quy định khác\nx"  # no repeated header: nothing may be dropped
    assert "Quy định khác" in join_article([_part(0, "1. một", 2), p])[0]


def test_long_article_is_windowed_around_the_matched_part():
    parts = [_part(i, f"{i + 1}. " + "x" * 100, parts=6) for i in range(6)]
    text, truncated = join_article(parts, focus_part=4, max_chars=300)
    assert truncated and len(text) <= 300 + len(HEADER) + 20
    assert "5. xxx" in text and "6. xxx" in text and "1. xxx" not in text
    assert text.startswith(HEADER) and "[...]" in text  # a gap is marked, the header is kept


def test_expandable_needs_more_than_one_part():
    assert expandable({"parts": 3}) and not expandable({"parts": 1}) and not expandable({}) and not expandable({"parts": "x"})


# ---------- retrieval wiring ----------

def _pt(pid, score, payload):
    return SimpleNamespace(id=pid, score=score, payload=payload)


@pytest.fixture
def world(monkeypatch):
    monkeypatch.setattr(settings, "SEARCH_MODE", "dense")
    monkeypatch.setattr(settings, "RERANK", False)
    monkeypatch.setattr(settings, "EXPAND_ARTICLES", True)
    monkeypatch.setattr(settings, "EXPAND_MAX_CHARS", 6000)
    monkeypatch.setattr(settings, "EXPAND_TOTAL_CHARS", 16000)
    monkeypatch.setattr(retrieval, "_encode_query", lambda q: ([0.0], ([], [])))
    single = {"doc_name": "399_2026_ND-CP_1", "article": "2", "text": "Điều 2. Ngắn\nnội dung", "part": 0, "parts": 1}
    monkeypatch.setattr(retrieval, "_query", lambda using, vec, limit, flt: [
        _pt("c2", 0.9, FOUR[2]), _pt("s", 0.8, single), _pt("c0", 0.7, FOUR[0]), _pt("c3", 0.6, FOUR[3])])
    fetched = []
    monkeypatch.setattr(retrieval, "fetch_article", lambda doc, art, limit=64: fetched.append((doc, art)) or list(FOUR))
    retrieval.clear_caches()
    return fetched


def test_hit_on_one_part_returns_the_whole_article_once(world):
    hits = retrieval.retrieve("điều 7 quy định gì", top_k=5)
    assert [h.payload["article"] for h in hits] == ["7", "2"]  # 3 parts of Điều 7 collapsed into one hit
    big = hits[0]
    assert big.payload["text"] == f"{HEADER}\n1. một\n2. hai\n3. ba\n4. bốn" and big.payload["expanded"]
    assert big.score == 0.9 and big.payload["match_text"].endswith("3. ba")  # best part's score and text
    assert hits[1].payload["text"].startswith("Điều 2.") and "expanded" not in hits[1].payload  # single chunk untouched
    assert world == [("399_2026_ND-CP_1", "7")]  # one fetch, none for the single-chunk Điều


def test_collapsing_duplicates_lets_other_articles_into_top_k(world):
    assert len(retrieval.retrieve("x", top_k=2)) == 2


def test_article_fetch_is_cached(world):
    retrieval.retrieve("điều 7", top_k=5)
    retrieval._result_cache.clear()  # force a second search, keep the article cache
    retrieval.retrieve("điều 7", top_k=5)
    assert len(world) == 1


def test_expansion_can_be_switched_off(world, monkeypatch):
    monkeypatch.setattr(settings, "EXPAND_ARTICLES", False)
    retrieval.clear_caches()
    hits = retrieval.retrieve("x", top_k=5)
    assert len(hits) == 4 and world == []


def test_failed_sibling_fetch_keeps_the_small_chunk(world, monkeypatch):
    def boom(doc, art, limit=64):
        raise retrieval.RetrievalError("qdrant down")
    monkeypatch.setattr(retrieval, "fetch_article", boom)
    retrieval.clear_caches()
    hits = retrieval.retrieve("x", top_k=5)
    assert hits[0].payload["text"].endswith("3. ba") and "expanded" not in hits[0].payload


def test_total_budget_limits_how_much_is_expanded(world, monkeypatch):
    monkeypatch.setattr(settings, "EXPAND_TOTAL_CHARS", 10)  # nothing fits: fall back to the matched part's own size
    retrieval.clear_caches()
    hits = retrieval.retrieve("x", top_k=5)
    assert hits[0].payload["truncated"] and len(hits[0].payload["text"]) < 60


def test_citation_snippet_shows_the_matched_part_and_names_the_article(world):
    hit = retrieval.retrieve("x", top_k=5)[0]
    c = rag.to_citation(1, hit)
    assert c.article == "7" and c.clause == "" and c.snippet.endswith("3. ba")


def test_prompt_contains_every_part_of_the_article(world):
    from app.prompts import build_user_prompt
    prompt = build_user_prompt("q", retrieval.retrieve("x", top_k=5)[:1])
    assert all(w in prompt for w in ("1. một", "2. hai", "3. ba", "4. bốn"))


def test_join_continuation_without_overlap_adds_no_line_break():
    a = _part(0, "Câu một. Câu hai", 2)
    b = _part(1, " và câu ba.", 2)
    b["cont"] = True
    assert join_article([a, b])[0] == f"{HEADER}\nCâu một. Câu hai và câu ba."


def test_points_without_part_are_ordered_by_clause_and_none_dropped():
    old = [{"clause": "10", "text": "Điều 3. T\nK10"}, {"clause": "2", "text": "Điều 3. T\nK2"}, {"clause": "1", "text": "Điều 3. T\nK1"}]
    text, _ = join_article(old)
    assert text.index("K1") < text.index("K2") < text.index("K10")
