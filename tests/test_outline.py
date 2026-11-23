from app.outline import build_outline


def _p(article, section=0, title="", chapter="", section_title=""):
    d = {"doc_name": "d", "article": article, "article_title": title, "chapter": chapter}
    if section:
        d["section"] = section
    if section_title:
        d["section_title"] = section_title
    return d


def test_counts_one_article_once_however_many_chunks_it_has():
    payloads = [_p("1"), _p("1"), _p("2"), _p("2"), _p("2"), _p("3")]
    out = build_outline("d", "Quyết định d", payloads)
    assert out["total_articles"] == 3 and out["section_count"] == 1 and not out["numbering_restarts"]
    assert [a["article"] for a in out["sections"][0]["articles"]] == ["1", "2", "3"]


def test_two_parts_with_restarting_numbers_are_counted_separately_and_summed():
    payloads = [_p(str(i)) for i in (1, 2, 3)] + [_p(str(i), 1, section_title="QUY CHẾ X") for i in range(1, 31)]
    out = build_outline("d", "Quyết định d", payloads)
    assert out["numbering_restarts"] and out["section_count"] == 2
    assert [s["count"] for s in out["sections"]] == [3, 30] and out["total_articles"] == 33
    assert out["sections"][1]["title"] == "QUY CHẾ X" and out["sections"][0]["title"] == "Phần chính"
    assert out["sections"][1]["first"] == "1" and out["sections"][1]["last"] == "30"


def test_articles_are_numbered_in_natural_order_with_letter_suffixes():
    out = build_outline("d", "d", [_p("10"), _p("2"), _p("5a"), _p("5")])
    assert [a["article"] for a in out["sections"][0]["articles"]] == ["2", "5", "5a", "10"]


def test_chapters_keep_reading_order_and_old_points_without_section_are_part_zero():
    out = build_outline("d", "d", [_p("3", chapter="Chương II"), _p("1", chapter="Chương I"), _p("2", chapter="Chương I")])
    assert out["sections"][0]["chapters"] == ["Chương I", "Chương II"] and out["sections"][0]["section"] == 0


def test_unknown_document_is_not_found():
    out = build_outline("nope", "nope", [])
    assert out["found"] is False and out["total_articles"] == 0 and out["sections"] == []
