"""Rerank vector candidates using lexical relevance and entity specificity.

Entity fan-out weighting and query-length sigmoid parameters follow the verified
Mem0 scoring implementation. BM25 statistics are local to the scoped candidate
pool and token normalization uses an English dictionary. The adopted video
contract uses a fixed divisor of 2.5, even when a score component is absent.
Entity matches change ordering but never introduce memories outside the pool.
"""

import math
import re
from collections import Counter

from simplemma import lemmatize

from .schema.memory_schema import SearchResult

EntityHit = tuple[str, float, list[str]]


def lemmas(text: str) -> list[str]:
    """Lowercase and lemmatize English word tokens, retaining term frequency.

    Unknown words, identifiers, and non-English text remain searchable as tokens;
    English dictionary lookup is not multilingual morphological analysis.
    """
    return [lemmatize(token, lang="en") for token in re.findall(r"\w+", text.lower())]


def _bm25(query: list[str], documents: list[list[str]]) -> list[float]:
    """Positive-IDF BM25 with k1=1.5, b=0.75, and query-adaptive sigmoid."""
    counts = [Counter(document) for document in documents]
    frequencies = Counter(term for document in counts for term in document)
    average_length = sum(map(len, documents)) / len(documents)
    if not average_length:
        return [0.0] * len(documents)
    scores = []
    for document, terms in zip(documents, counts):
        score = 0.0
        length_factor = 1.5 * (0.25 + 0.75 * len(document) / average_length)
        for term in set(query):
            frequency = terms[term]
            if frequency:
                idf = math.log1p(
                    (len(documents) - frequencies[term] + 0.5) / (frequencies[term] + 0.5)
                )
                score += idf * frequency * 2.5 / (frequency + length_factor)
        scores.append(score)
    midpoint, steepness = next(
        (midpoint, steepness)
        for maximum, midpoint, steepness in [
            (3, 5.0, 0.7),
            (6, 7.0, 0.6),
            (9, 9.0, 0.5),
            (15, 10.0, 0.5),
            (math.inf, 12.0, 0.5),
        ]
        if len(query) <= maximum
    )
    return [
        1.0 / (1.0 + math.exp(-steepness * (score - midpoint))) if score > 0 else 0.0
        for score in scores
    ]


def rerank(
    query: str,
    candidates: list[SearchResult],
    entity_hits: list[EntityHit],
) -> list[tuple[SearchResult, dict[str, float]]]:
    """Return candidates with auditable components and deterministic final ranking.

    Vector and keyword scores lie in [0,1]. Matching entities require cosine
    similarity >=0.5 and contribute ``0.5*similarity/(1+0.001*(link_count-1)^2)``.
    The strongest entity contribution is retained, bounded by 0.5. The final
    score sums the three components and divides by 2.5. Fan-out includes links
    outside the candidate pool, preserving the penalty for common entities.
    """
    if not candidates:
        return []
    if any(not math.isfinite(candidate.score) for candidate in candidates):
        raise ValueError("Candidate scores must be finite")
    candidate_ids = {candidate.memory.id for candidate in candidates}
    boosts = dict.fromkeys(candidate_ids, 0.0)
    for name, similarity, linked_ids in entity_hits:
        links = set(linked_ids)
        if not name.strip() or not math.isfinite(similarity) or similarity < 0.5 or not links:
            continue
        weight = 0.5 * min(1.0, similarity) / (1.0 + 0.001 * (len(links) - 1) ** 2)
        for memory_id in links & candidate_ids:
            boosts[memory_id] = max(boosts[memory_id], weight)
    keywords = _bm25(
        lemmas(query),
        [candidate.memory.keywords or lemmas(candidate.memory.content) for candidate in candidates],
    )
    ranked = []
    for candidate, keyword in zip(candidates, keywords):
        vector = min(1.0, max(0.0, candidate.score))
        boost = boosts[candidate.memory.id]
        components = {
            "vector_score": vector,
            "keyword_score": keyword,
            "entity_boost": boost,
            "score": (vector + keyword + boost) / 2.5,
        }
        ranked.append((candidate, components))
    return sorted(ranked, key=lambda item: (-item[1]["score"], item[0].memory.id))
