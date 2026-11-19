"""Baseline metrics. Run BEFORE and AFTER every improvement (Phase 3 onwards) and compare.

  python -m scripts.baseline --label baseline            # retrieval only (no LLM key needed)
  python -m scripts.baseline --label baseline --with-llm # + end-to-end latency, grounding, citation accuracy
  python -m scripts.baseline --label phase3 --matrix     # compare dense / hybrid / dense+rerank / hybrid+rerank
  python -m scripts.baseline --mode dense --no-rerank    # one specific configuration

question file format (eval/questions.jsonl), one JSON per line:
  {"question": "...", "expected": [{"doc": "399_2026", "article": "3"}]}      # 'doc' = prefix of doc_name
  {"question": "...", "expected": [], "should_refuse": true}                 # out-of-scope question
"""
import argparse
import asyncio
import datetime
import json
import time
from pathlib import Path

from app import rag, retrieval, settings


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(pct / 100 * (len(ordered) - 1))))]


def matches(payload: dict, expected: list[dict]) -> bool:
    return any(
        str(payload.get("doc_name", "")).startswith(e["doc"]) and str(payload.get("article")) == str(e["article"])
        for e in expected
    )


def first_rank(hits, expected: list[dict]) -> int | None:
    for rank, h in enumerate(hits, 1):
        if matches(h.payload, expected):
            return rank
    return None


async def search(question: str, k: int, mode: str, rerank: bool):
    return await asyncio.to_thread(retrieval.retrieve, question, k, None, mode, rerank)


async def run(args, mode: str, rerank: bool) -> str:
    threshold = settings.RERANK_MIN_SCORE if rerank else settings.MIN_SCORE
    await search("khởi động", 1, mode, rerank)  # warm-up: model loading must not count as latency
    rows = [json.loads(l) for l in Path(args.questions).read_text(encoding="utf-8").splitlines() if l.strip()]
    answerable = [r for r in rows if not r.get("should_refuse")]
    unanswerable = [r for r in rows if r.get("should_refuse")]

    ranks, ret_ms, top1_ok, top1_bad = [], [], [], []
    misses = []
    for r in answerable:
        t0 = time.perf_counter()
        hits = await search(r["question"], args.k, mode, rerank)
        ret_ms.append((time.perf_counter() - t0) * 1000)
        rank = first_rank(hits, r["expected"])
        ranks.append(rank)
        if hits:
            top1_ok.append(max(h.score for h in hits))
        if rank is None or rank > 3:
            misses.append((r["question"], rank, [(h.payload["doc_name"][:3], h.payload["article"]) for h in hits[:3]]))
    for r in unanswerable:
        hits = await search(r["question"], args.k, mode, rerank)
        top1_bad.append(max((h.score for h in hits), default=0.0))

    n = len(answerable)
    recall = lambda k: sum(1 for x in ranks if x is not None and x <= k) / n if n else 0.0
    mrr = sum(1 / x for x in ranks if x) / n if n else 0.0
    rejected = sum(1 for s in top1_bad if s < threshold)

    lines = [
        f"### {args.label} [{mode}{'+rerank' if rerank else ''}] - {datetime.datetime.now():%Y-%m-%d %H:%M} (k={args.k}, threshold={threshold})",
        "",
        "| Metric | Value |", "|---|---|",
        f"| Answerable questions | {n} |",
        f"| Recall@1 | {recall(1):.2f} |",
        f"| Recall@3 | {recall(3):.2f} |",
        f"| Recall@{args.k} | {recall(args.k):.2f} |",
        f"| MRR | {mrr:.3f} |",
        f"| Retrieval latency p50 / p95 (ms) | {percentile(ret_ms, 50):.0f} / {percentile(ret_ms, 95):.0f} |",
        f"| Out-of-scope questions rejected by threshold | {rejected}/{len(unanswerable)} |",
        f"| Lowest top-1 score, answerable | {min(top1_ok):.3f} |" if top1_ok else "",
        f"| Highest top-1 score, out-of-scope | {max(top1_bad):.3f} |" if top1_bad else "",
    ]

    if args.with_llm:
        e2e, grounded, cite_ok, refused_ok = [], 0, 0, 0
        for r in answerable:
            res = await rag.answer(r["question"], args.k)
            e2e.append(res.latency_ms.total_ms)
            grounded += res.grounded
            cite_ok += any(matches({"doc_name": c.doc_name, "article": c.article}, r["expected"]) for c in res.citations)
        for r in unanswerable:
            res = await rag.answer(r["question"], args.k)
            refused_ok += (not res.grounded)
        lines += [
            f"| End-to-end latency p50 / p95 (ms) | {percentile(e2e, 50):.0f} / {percentile(e2e, 95):.0f} |",
            f"| Answers with at least one citation | {grounded}/{n} |",
            f"| Answers citing the expected Article | {cite_ok}/{n} |",
            f"| Out-of-scope correctly refused | {refused_ok}/{len(unanswerable)} |",
        ]

    lines = [l for l in lines if l != ""]
    lines.insert(2, "")
    if misses:
        lines += ["", "Questions where the expected Article was not in the top 3:"]
        lines += [f"- {q}  (rank={rk}, got={got})" for q, rk, got in misses]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="baseline")
    ap.add_argument("--questions", default="eval/questions.jsonl")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--with-llm", action="store_true")
    ap.add_argument("--mode", choices=["dense", "hybrid"], default=None, help="default: SEARCH_MODE")
    ap.add_argument("--rerank", dest="rerank", action="store_true", default=None)
    ap.add_argument("--no-rerank", dest="rerank", action="store_false")
    ap.add_argument("--matrix", action="store_true", help="run all 4 configurations")
    ap.add_argument("--out", default="eval/baseline_results.md")
    args = ap.parse_args()

    if args.matrix:
        configs = [("dense", False), ("hybrid", False), ("dense", True), ("hybrid", True)]
    else:
        configs = [(args.mode or settings.SEARCH_MODE, settings.RERANK if args.rerank is None else args.rerank)]
    for mode, rerank in configs:
        report = asyncio.run(run(args, mode, rerank))
        print(report)
        with open(args.out, "a", encoding="utf-8") as f:
            f.write(report + "\n")
    print(f"Appended to {args.out}")


if __name__ == "__main__":
    main()
