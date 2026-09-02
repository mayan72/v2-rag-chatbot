"""
Semantic Retriever

Responsibilities
----------------
1. Load local embedding model once.
2. Connect to persistent ChromaDB.
3. Convert user query to embedding.
4. Retrieve top-k similar chunks.
5. Apply similarity threshold.
6. Remove duplicate chunks.
7. Build context.
8. Return structured retrieval result.

This module DOES NOT call the LLM.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Any

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from config import (
    EMBEDDING_MODEL,
    CHROMA_COLLECTION_NAME,
    CHROMA_DB_PATH,
    TOP_K_RESULTS,
    SIMILARITY_THRESHOLD,
    MIN_CHUNK_SIMILARITY,
    MAX_CONTEXT_CHUNKS,
)
from debug_trace import dbg

logger = logging.getLogger(__name__)


# ============================================================
# Dataclasses
# ============================================================

@dataclass
class RetrievedChunk:
    """
    Represents one retrieved document chunk.
    """

    content: str

    metadata: Dict[str, Any]

    similarity: float


@dataclass
class RetrievalResult:
    """
    Returned by retrieve().
    """

    should_answer: bool

    confidence: float

    max_similarity: float

    retrieval_time_ms: float

    chunks: List[RetrievedChunk] = field(default_factory=list)

    context: str = ""

    sources: List[Dict[str, Any]] = field(default_factory=list)


# ============================================================
# Retriever
# ============================================================

class SemanticRetriever:

    def __init__(self):

        logger.info("Loading embedding model...")

        self.embedding_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={
                "device": "cpu"
            },
            encode_kwargs={
                "normalize_embeddings": True
            },
        )

        logger.info("Connecting to ChromaDB...")

        self.vector_db = Chroma(
            collection_name=CHROMA_COLLECTION_NAME,
            embedding_function=self.embedding_model,
            persist_directory=CHROMA_DB_PATH,
        )

        logger.info("Retriever initialized successfully.")

    def _distance_to_cosine(self, distance: float) -> float:
        d = float(distance)
        similarity = 1.0 - (d * d) / 2.0
        return max(0.0, min(1.0, similarity))


    # ============================================================
    # Private Helpers
    # ============================================================

    def _calculate_confidence(self, similarities: List[float]) -> float:
        """
        Calculate an overall confidence score for the retrieval.

        Strategy:
        - No matches -> 0.0
        - Average of Top-3 similarity scores
        - Clamp between 0 and 1

        Args:
            similarities: List of similarity scores.

        Returns:
            float: Confidence score between 0.0 and 1.0
        """

        if not similarities:
            return 0.0

        top_scores = sorted(similarities, reverse=True)[:3]

        confidence = sum(top_scores) / len(top_scores)

        confidence = max(0.0, min(confidence, 1.0))

        logger.debug(
            "Confidence calculated | top_scores=%s | confidence=%.4f",
            top_scores,
            confidence,
        )

        return round(confidence, 4)
    

    def _deduplicate_chunks(
        self,
        chunks: List[RetrievedChunk],
    ) -> List[RetrievedChunk]:
        """
        Remove duplicate chunks while preserving order.

        Chunks are considered duplicates if their content is identical
        after trimming whitespace.

        Args:
            chunks: Retrieved chunks.

        Returns:
            List[RetrievedChunk]
        """

        unique_chunks: List[RetrievedChunk] = []

        seen_contents = set()

        for chunk in chunks:

            normalized = chunk.content.strip()

            if normalized in seen_contents:
                continue

            seen_contents.add(normalized)

            unique_chunks.append(chunk)

        logger.debug(
            "Deduplicated chunks | before=%d | after=%d",
            len(chunks),
            len(unique_chunks),
        )

        return unique_chunks
    
    def _extract_sources(
        self,
        chunks: List[RetrievedChunk],
    ) -> List[Dict[str, Any]]:

        sources = []

        for chunk in chunks:

            metadata = dict(chunk.metadata)
            metadata["similarity"] = round(chunk.similarity, 4)
            metadata["content"] = chunk.content
            sources.append(metadata)

        return sources
    
    def _build_context(
        self,
        chunks: List[RetrievedChunk],
    ) -> str:
        """
        Build the context that will be sent to the LLM.

        Each retrieved chunk is separated clearly so the model
        can distinguish between different sources.

        Args:
            chunks: List of retrieved chunks.

        Returns:
            Context string.
        """

        if not chunks:
            return ""

        context_parts = []

        for index, chunk in enumerate(chunks, start=1):

            metadata = chunk.metadata or {}
            source_name = metadata.get("document_name", "unknown")
            row_number = metadata.get("row_number", "")

            extra_fields = []
            for key, value in metadata.items():
                if key in {
                    "document_name",
                    "document_id",
                    "source_type",
                    "row_number",
                    "similarity",
                    "content",
                }:
                    continue
                if value is None:
                    continue
                extra_fields.append(f"{key}: {value}")

            extras = "\n".join(extra_fields)
            extras_block = f"\n{extras}\n" if extras else "\n"

            context_parts.append(
                f"""
==============================
DOCUMENT {index}
==============================
Source: {source_name} | Row: {row_number}
Similarity: {chunk.similarity:.4f}
{extras_block}
Content:
{chunk.content.strip()}
"""
            )

        context = "\n".join(context_parts)

        logger.debug(
            "Context built successfully | documents=%d | characters=%d",
            len(chunks),
            len(context),
        )

        return context
        
        # ============================================================
    # Public API
    # ============================================================

    def retrieve(
        self,
        question: str,
    ) -> RetrievalResult:
        """
        Retrieve relevant chunks for a question.

        Parameters
        ----------
        question : str

        Returns
        -------
        RetrievalResult
        """

        start = time.perf_counter()

        # -------------------------------------------------------
        # Search Chroma
        # -------------------------------------------------------

        results = self.vector_db.similarity_search_with_score(
            query=question,
            k=TOP_K_RESULTS,
        )

        retrieval_time = (time.perf_counter() - start) * 1000

        chunks = []

        similarities = []

        # -------------------------------------------------------
        # Convert LangChain Documents
        # -------------------------------------------------------

        for document, distance in results:

            similarity = self._distance_to_cosine(distance)

            chunk = RetrievedChunk(
                content=document.page_content,
                metadata=document.metadata,
                similarity=similarity,
            )

            chunks.append(chunk)

        chunks = self._deduplicate_chunks(chunks)

        chunks = [
            chunk
            for chunk in chunks
            if chunk.similarity >= MIN_CHUNK_SIMILARITY
        ]

        chunks = chunks[:MAX_CONTEXT_CHUNKS]

        similarities = [chunk.similarity for chunk in chunks]

        confidence = self._calculate_confidence(similarities)

        max_similarity = max(similarities) if similarities else 0.0

        should_answer = (
            bool(chunks)
            and max_similarity >= SIMILARITY_THRESHOLD
        )

        # -------------------------------------------------------
        # Context
        # -------------------------------------------------------

        context = self._build_context(chunks)

        # -------------------------------------------------------
        # Sources
        # -------------------------------------------------------

        sources = self._extract_sources(chunks)

        logger.debug(
            "Retrieved %d chunks | "
            "Confidence=%.4f | "
            "Max Similarity=%.4f | "
            "Time=%.2f ms",
            len(chunks),
            confidence,
            max_similarity,
            retrieval_time,
        )
        dbg(
            "RETRIEVE_SCORES",
            k=len(results),
            kept=len(chunks),
            should_answer=should_answer,
            max_similarity=max_similarity,
            scores=[
                {
                    "sim": round(chunk.similarity, 4),
                    "preview": (chunk.content or "")[:180],
                }
                for chunk in chunks
            ],
        )

        return RetrievalResult(
            should_answer=should_answer,
            confidence=confidence,
            max_similarity=max_similarity,
            # retrieval_time_ms=round(retrieval_time, 2),
            retrieval_time_ms = round(retrieval_time / 1000, 3),
            chunks=chunks,
            context=context,
            sources=sources,
        )