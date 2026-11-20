from app.cache import TTLCache, normalize_query


def test_lru_evicts_least_recently_used():
    c = TTLCache(maxsize=2)
    c.set("a", 1)
    c.set("b", 2)
    c.get("a")  # a is now fresher than b
    c.set("c", 3)
    assert c.get("b") is None and c.get("a") == 1 and c.get("c") == 3


def test_entries_expire_after_ttl():
    now = [0.0]
    c = TTLCache(maxsize=5, ttl=10, clock=lambda: now[0])
    c.set("k", "v")
    now[0] = 9
    assert c.get("k") == "v"
    now[0] = 11
    assert c.get("k") is None and len(c) == 0


def test_maxsize_zero_disables_the_cache():
    c = TTLCache(maxsize=0)
    c.set("k", "v")
    assert c.get("k") is None and not c.enabled


def test_stats_count_hits_and_misses():
    c = TTLCache(maxsize=2)
    c.get("x")
    c.set("x", 1)
    c.get("x")
    assert c.stats() == {"size": 1, "hits": 1, "misses": 1}


def test_normalize_query_ignores_case_spacing_and_trailing_punctuation():
    assert normalize_query("  Mức   LƯƠNG tối thiểu? ") == normalize_query("mức lương tối thiểu")
