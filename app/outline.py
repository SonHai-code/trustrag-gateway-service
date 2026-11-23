"""Document outline: how many Điều / chapters a document has, per part.

Pure function over the payloads of every chunk of ONE document (no I/O). Counting is a database fact, not something a
language model should estimate from the few passages a similarity search returns.

A document can hold several numbered parts, e.g. a Quyết định (Điều 1-3) with an attached Quy chế (Điều 1-30). The
ingestion chunker gives every chunk a `section` number (0 = the main text, 1, 2, ... = parts that restart at Điều 1)
and, when it can find one, a `section_title` ("QUY CHẾ ..."). Older points have no `section`: they all count as part 0.
"""
import re

from app.expand import _num

_ART = re.compile(r"^(\d+)([a-zA-Z]*)$")


def _article_order(article: str) -> tuple[int, str]:
    m = _ART.match(str(article).strip())
    return (int(m.group(1)), m.group(2).lower()) if m else (10**6, str(article))


def build_outline(doc_name: str, label: str, payloads: list[dict]) -> dict:
    sections: dict[int, dict] = {}
    for p in payloads:
        art = str(p.get("article", "")).strip()
        if not art:
            continue
        sec = sections.setdefault(_num(p.get("section")), {"title": "", "articles": {}})
        if p.get("section_title") and not sec["title"]:
            sec["title"] = str(p["section_title"])
        sec["articles"].setdefault(art, {"article": art, "title": p.get("article_title", ""), "chapter": p.get("chapter", "")})
    out = []
    for n in sorted(sections):
        sec = sections[n]
        arts = sorted(sec["articles"].values(), key=lambda a: _article_order(a["article"]))
        chapters: list[str] = []
        for a in arts:
            if a["chapter"] and a["chapter"] not in chapters:
                chapters.append(a["chapter"])
        out.append({"section": n, "title": sec["title"] or ("Phần chính" if n == 0 else f"Phần {n + 1}"),
                    "count": len(arts), "first": arts[0]["article"] if arts else "", "last": arts[-1]["article"] if arts else "",
                    "chapters": chapters, "articles": arts})
    return {"doc_name": doc_name, "doc_label": label, "found": bool(out),
            "total_articles": sum(s["count"] for s in out), "section_count": len(out),
            "numbering_restarts": len(out) > 1, "sections": out}
