"""
Lightweight reranker over Chroma chunks.

Does not replace Chroma. Scores keyword/metric/year overlap on top of
existing similarity.
"""

from __future__ import annotations

import re
from typing import List

from rag.query_analyzer import QueryAnalysis


def _tokens(text: str) -> set:
    return set(re.findall(r"[a-z0-9]+", (text or "").lower()))


def rerank_chunks(query: str, chunks: List, analysis: QueryAnalysis | None = None) -> List:
    if not chunks:
        return chunks

    query_tokens = _tokens(query)
    extra = set()
    if analysis:
        if analysis.metric:
            extra.add(analysis.metric.lower())
        if analysis.start_period:
            extra.add(analysis.start_period)
        if analysis.end_period:
            extra.add(analysis.end_period)

    scored = []
    for chunk in chunks:
        content = getattr(chunk, "content", "") or ""
        chunk_tokens = _tokens(content)
        overlap = 0.0
        if query_tokens:
            overlap = len(query_tokens & chunk_tokens) / len(query_tokens)
        bonus = 0.0
        if extra:
            bonus = len(extra & chunk_tokens) / len(extra)
        similarity = float(getattr(chunk, "similarity", 0.0) or 0.0)
        score = (0.55 * similarity) + (0.25 * overlap) + (0.20 * bonus)
        setattr(chunk, "rerank_score", round(score, 4))
        scored.append((score, chunk))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [item[1] for item in scored]
