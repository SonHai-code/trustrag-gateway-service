"""Small-to-big, step 2: rebuild the whole Điều from the small chunks the vector search found.

Pure functions (no I/O). Chunks of one Điều carry `part` / `parts` / `overlap` (see the ingestion chunker):
every part repeats the "Điều N. Title" line; a continuation piece (`cont`) carries on mid-text and may repeat
`overlap` characters of the previous one.
"""

GAP = "\n[...]\n"


def expandable(payload: dict) -> bool:
    """Only chunks of an Điều that was cut into several parts need the rest; older points have no `parts` field."""
    try:
        return int(payload.get("parts") or 1) > 1
    except (TypeError, ValueError):
        return False


def _num(value, default=0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def ordered(payloads: list[dict]) -> list[dict]:
    """Parts in reading order, one per `part` number (a re-ingestion race could leave duplicates).
    Points without `part` (very old index) are ordered by their clause number instead."""
    if any("part" not in p for p in payloads):
        return sorted(payloads, key=lambda p: _num(p.get("clause"), 10**6))
    seen: dict[int, dict] = {}
    for p in sorted(payloads, key=lambda p: _num(p.get("part"))):
        seen.setdefault(_num(p.get("part")), p)
    return list(seen.values())


def _pieces(parts: list[dict]) -> tuple[str, list[tuple[str, bool]]]:
    """-> (header, [(text_without_header_and_overlap, continues_previous_text)])."""
    header = ""
    out: list[tuple[str, bool]] = []
    for i, p in enumerate(parts):
        first, _, rest = str(p.get("text", "")).partition("\n")
        if i == 0:
            header = first
        elif first != header:  # not a repeated header: it was real content
            rest = str(p.get("text", ""))
        rep = _num(p.get("overlap"))
        out.append((rest[rep:] if i else rest, i > 0 and (bool(rep) or bool(p.get("cont")))))
    return header, out


def join_article(payloads: list[dict], focus_part: int | None = None, max_chars: int | None = None) -> tuple[str, bool]:
    """Whole Điều text from its chunks. With `max_chars`, keep a window of parts around `focus_part` (the part the
    search matched) instead of cutting the text blindly. Returns (text, truncated)."""
    parts = ordered(payloads)
    if not parts:
        return "", False
    header, pieces = _pieces(parts)

    def render(idx: list[int]) -> str:
        out, prev = header, None
        for i in idx:
            text, cont = pieces[i]
            if prev is None:
                out += ("\n" if i == 0 else GAP) + text
            elif i != prev + 1:
                out += GAP + text
            else:
                out += ("" if cont else "\n") + text
            prev = i
        if idx[-1] != len(pieces) - 1:
            out += GAP.rstrip("\n")
        return out

    everything = list(range(len(pieces)))
    full = render(everything)
    if max_chars is None or len(full) <= max_chars:
        return full, False
    # too long: grow a window outwards from the matched part, next part first
    nums = [_num(p.get("part")) for p in parts]
    start = nums.index(focus_part) if focus_part in nums else 0
    chosen, lo, hi = [start], start, start
    size = len(render(chosen))
    while True:
        grew = False
        for cand in (hi + 1, lo - 1):
            if 0 <= cand < len(pieces) and cand not in chosen:
                extra = len(pieces[cand][0]) + len(GAP)  # worst case: a gap marker per part
                if size + extra <= max_chars:
                    chosen.append(cand)
                    lo, hi = min(lo, cand), max(hi, cand)
                    size += extra
                    grew = True
        if not grew:
            break
    return render(sorted(chosen)), True
