import math

import pytest
from mem_alive.retrieval import lemmas, rerank
from mem_alive.schema.memory_schema import Memory, SearchResult


def candidate(memory_id: str, content: str, score: float = 0.8) -> SearchResult:
    return SearchResult(Memory(memory_id, content, [1.0], {}, "semantic", "project"), score)


def test_english_lemmas_and_coding_identifiers():
    assert lemmas("RUNNING tests, memories! src/main.py foo_bar") == [
        "run",
        "test",
        "memory",
        "src",
        "main",
        "py",
        "foo_bar",
    ]


def test_keyword_match_breaks_vector_tie_and_matches_inflections():
    hits = rerank(
        "running tests",
        [
            candidate("irrelevant", "Deploy application"),
            candidate("target", "Run test"),
        ],
        [],
    )
    assert [hit.memory.id for hit, _ in hits] == ["target", "irrelevant"]
    # Two equal-length documents, two query terms present in one document:
    # raw BM25 = 2*ln(2), then the short-query sigmoid (midpoint=5, slope=.7).
    expected_keyword = 1 / (1 + math.exp(-0.7 * (2 * math.log(2) - 5)))
    assert hits[0][1] == pytest.approx(
        {
            "vector_score": 0.8,
            "keyword_score": expected_keyword,
            "entity_boost": 0.0,
            "score": (0.8 + expected_keyword) / 2.5,
        }
    )
    assert hits[1][1]["keyword_score"] == 0.0


def test_bm25_length_normalization_and_term_frequency():
    hits = rerank(
        "pytest",
        [
            candidate("short", "pytest"),
            candidate("long", "pytest long unrelated deployment command details"),
            candidate("repeat", "pytest pytest pytest"),
        ],
        [],
    )
    scores = {hit.memory.id: parts["keyword_score"] for hit, parts in hits}
    assert 1 > scores["repeat"] > scores["short"] > scores["long"] > 0


def test_entity_specificity_uses_all_links_and_never_adds_candidates():
    hits = rerank(
        "",
        [candidate("rare", "a"), candidate("common", "b")],
        [
            ("specific", 0.8, ["rare"]),
            ("generic", 0.8, ["common", "external-a", "external-b", "external-c"]),
            ("phantom", 1.0, ["missing"]),
        ],
    )
    assert [hit.memory.id for hit, _ in hits] == ["rare", "common"]
    assert hits[0][1]["entity_boost"] == 0.4
    assert hits[1][1]["entity_boost"] == pytest.approx(0.4 / 1.009)


def test_entity_duplicates_nonpositive_and_boost_cap():
    hits = rerank(
        "",
        [candidate("a", "")],
        [
            ("entity", 0.6, ["a", "a"]),
            ("ENTITY", 0.6, ["a"]),
            ("negative", -1, ["a"]),
            ("zero", 0, ["a"]),
            ("nonfinite", math.nan, ["a"]),
        ],
    )
    assert hits[0][1]["entity_boost"] == 0.3
    boosted = rerank(
        "",
        [candidate("a", "", 2)],
        [
            ("first", 5, ["a"]),
            ("second", 1, ["a"]),
        ],
    )
    assert boosted[0][1]["entity_boost"] == 0.5
    assert boosted[0][1]["vector_score"] == 1.0
    assert 0 <= boosted[0][1]["score"] <= 1


def test_empty_candidates_empty_documents_and_stable_ties():
    assert rerank("query", [], [("x", 1, ["phantom"])]) == []
    hits = rerank("query", [candidate("z", "", -1), candidate("a", "", -1)], [])
    assert [hit.memory.id for hit, _ in hits] == ["a", "z"]
    assert all(parts["score"] == 0 for _, parts in hits)
    with pytest.raises(ValueError, match="finite"):
        rerank("query", [candidate("bad", "", math.inf)], [])


def test_entity_match_threshold_and_strongest_match_not_sum():
    below = rerank("", [candidate("a", "")], [("near", 0.499, ["a"])])
    assert below[0][1]["entity_boost"] == 0
    at_boundary = rerank("", [candidate("a", "")], [("near", 0.5, ["a"])])
    assert at_boundary[0][1]["entity_boost"] == 0.25
    strongest = rerank(
        "",
        [candidate("a", "")],
        [
            ("first", 0.8, ["a"]),
            ("second", 0.7, ["a"]),
        ],
    )
    assert strongest[0][1]["entity_boost"] == 0.4
    assert strongest[0][1]["score"] == pytest.approx(0.48)


@pytest.mark.parametrize(
    "terms,midpoint,slope",
    [
        (3, 5, 0.7),
        (4, 7, 0.6),
        (6, 7, 0.6),
        (7, 9, 0.5),
        (9, 9, 0.5),
        (10, 10, 0.5),
        (15, 10, 0.5),
        (16, 12, 0.5),
    ],
)
def test_query_length_sigmoid_parameters(terms, midpoint, slope):
    query = "pytest " + "unmatched " * (terms - 1)
    hits = rerank(query, [candidate("match", "pytest"), candidate("other", "deployment")], [])
    expected = 1 / (1 + math.exp(-slope * (math.log(2) - midpoint)))
    assert hits[0][1]["keyword_score"] == pytest.approx(expected)
    assert hits[1][1]["keyword_score"] == 0


def test_persisted_keyword_tokens_are_used():
    target = candidate("target", "No content match")
    target.memory.keywords = ["pytest"]
    hits = rerank("pytest", [target, candidate("other", "deployment")], [])
    expected = 1 / (1 + math.exp(-0.7 * (math.log(2) - 5)))
    assert hits[0][0].memory.id == "target"
    assert hits[0][1]["keyword_score"] == pytest.approx(expected)
