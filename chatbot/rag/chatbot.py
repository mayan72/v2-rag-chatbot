"""
Production RAG Chatbot

Responsibilities
----------------
1. Receive user question
2. Retrieve relevant documents
3. Build prompt
4. Call Grok API
5. Return structured response
6. Log everything

This class NEVER directly interacts with Chroma.
All retrieval happens through SemanticRetriever.
"""

from __future__ import annotations
from logger.run_logger import RunLogger
import logging
import time
from datetime import datetime
import uuid
from typing import Dict, Any
from logger.cost_calculator import CostCalculator
from logger.run_logger import RunLogger
from config import (
    TOP_K_RESULTS,
    SIMILARITY_THRESHOLD,
    LLM_TEMPERATURE,
)
from llm.llm_factory import LLMFactory
from rag.retriever import SemanticRetriever
from rag.prompt_builder import PromptBuilder
from rag.hybrid_qa import HybridQAEngine
from debug_trace import dbg
from logger.console import (
    configure_logging,
    document_names,
    format_filters,
    qlog,
)

logger = logging.getLogger(__name__)


class RAGChatbot:
    """
    Production-ready RAG chatbot.
    """

    def __init__(self):

        configure_logging()

        # -------------------------------------------------------
        # Retriever
        # -------------------------------------------------------

        self.retriever = SemanticRetriever()

        # -------------------------------------------------------
        # Prompt Builder
        # -------------------------------------------------------

        self.prompt_builder = PromptBuilder()
        self.run_logger = RunLogger()
        self.hybrid_qa = None

        # -------------------------------------------------------
        # Grok Client
        # -------------------------------------------------------

        # self.client = OpenAI(
        #     api_key=GROK_API_KEY,
        #     base_url=GROK_BASE_URL,
        # )
        # self.client = genai.Client(
        #     api_key=GOOGLE_API_KEY
        # )
        self.llm = LLMFactory.create()
        self.hybrid_qa = HybridQAEngine(llm=self.llm)

        self.cost_calculator = CostCalculator()

        qlog("CHATBOT", status="ready")

        # ============================================================
    # Private Methods
    # ============================================================

    
        # ============================================================

    def ask(
        self,
        question: str,
    ) -> Dict[str, Any]:
        """
        Execute the complete RAG pipeline.
        """

        overall_start = time.perf_counter()

        request_id = str(uuid.uuid4())
        qlog("QUESTION", text=question)
        dbg("ASK_START", request_id=request_id, question=question)

        # -------------------------------------------------------
        # Step 0 : Structured table QA (counts / sums / filters)
        # -------------------------------------------------------

        structured = self.hybrid_qa.answer(question)
        dbg(
            "STEP0_STRUCTURED_RESULT",
            request_id=request_id,
            has_result=bool(structured),
            matched=getattr(structured, "matched", None),
            answer=getattr(structured, "answer", None),
            operation=getattr(structured, "operation", None),
            row_count=getattr(structured, "row_count", None),
            table_id=getattr(structured, "table_id", None),
            document_name=getattr(structured, "document_name", None),
            filters=getattr(structured, "filters", None),
            source_count=len(getattr(structured, "sources", None) or []),
        )

        if structured and structured.matched:

            total_time = (
                time.perf_counter() - overall_start
            ) * 1000

            log_payload = {
                "status": "SUCCESS",
                "timestamp": datetime.now().isoformat(),
                "request_id": request_id,
                "provider": "structured",
                "model": "table-engine",
                "question": question,
                "answer": structured.answer,
                "confidence": 1.0,
                "max_similarity": 1.0,
                "should_answer": True,
                "chunks_retrieved": len(structured.sources or []),
                "context_length": 0,
                "retrieval_time_ms": 0,
                "llm_time_ms": 0,
                "llm_provider_latency_ms": 0,
                "total_time_ms": round(total_time, 2),
                "retrieval_threshold": SIMILARITY_THRESHOLD,
                "top_k": TOP_K_RESULTS,
                "temperature": LLM_TEMPERATURE,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "input_cost": 0,
                "output_cost": 0,
                "embedding_cost": 0,
                "total_cost": 0,
                "sources": structured.sources or [],
                "error": "",
            }

            qlog(
                "RESULT",
                path="structured",
                documents=structured.document_name,
                operation=structured.operation,
                rows=structured.row_count,
                filters=format_filters(structured.filters),
                answer=(structured.answer or "")[:240],
                time_ms=round(total_time, 2),
            )

            self.run_logger.log_success(log_payload)

            return {
                "answer": structured.answer,
                "confidence": 1.0,
                "provider": "structured",
                "model": "table-engine",
                "sources": structured.sources or [],
                "retrieval_time_ms": 0,
                "llm_time_ms": 0,
                "llm_provider_latency_ms": 0,
                "total_time_ms": round(total_time, 2),
                "chunks_retrieved": len(structured.sources or []),
                "context_length": 0,
                "retrieval_threshold": SIMILARITY_THRESHOLD,
                "top_k": TOP_K_RESULTS,
                "temperature": LLM_TEMPERATURE,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "cost": 0,
            }

        # -------------------------------------------------------
        # Step 0 : Structured table QA (counts / sums / filters)
        # -------------------------------------------------------

        structured = self.hybrid_qa.answer(question)

        if structured and structured.matched:

            total_time = (
                time.perf_counter() - overall_start
            ) * 1000

            log_payload = {
                "status": "SUCCESS",
                "timestamp": datetime.now().isoformat(),
                "request_id": request_id,
                "provider": "structured",
                "model": "table-engine",
                "question": question,
                "answer": structured.answer,
                "confidence": 1.0,
                "max_similarity": 1.0,
                "should_answer": True,
                "chunks_retrieved": len(structured.sources or []),
                "context_length": 0,
                "retrieval_time_ms": 0,
                "llm_time_ms": 0,
                "llm_provider_latency_ms": 0,
                "total_time_ms": round(total_time, 2),
                "retrieval_threshold": SIMILARITY_THRESHOLD,
                "top_k": TOP_K_RESULTS,
                "temperature": LLM_TEMPERATURE,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "input_cost": 0,
                "output_cost": 0,
                "embedding_cost": 0,
                "total_cost": 0,
                "sources": structured.sources or [],
                "error": "",
            }

            qlog(
                "RESULT",
                path="structured",
                documents=structured.document_name,
                operation=structured.operation,
                rows=structured.row_count,
                filters=format_filters(structured.filters),
                answer=(structured.answer or "")[:240],
                time_ms=round(total_time, 2),
            )

            self.run_logger.log_success(log_payload)

            return {
                "answer": structured.answer,
                "confidence": 1.0,
                "provider": "structured",
                "model": "table-engine",
                "sources": structured.sources or [],
                "retrieval_time_ms": 0,
                "llm_time_ms": 0,
                "llm_provider_latency_ms": 0,
                "total_time_ms": round(total_time, 2),
                "chunks_retrieved": len(structured.sources or []),
                "context_length": 0,
                "retrieval_threshold": SIMILARITY_THRESHOLD,
                "top_k": TOP_K_RESULTS,
                "temperature": LLM_TEMPERATURE,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "cost": 0,
            }

        # -------------------------------------------------------
        # Step 1 : Retrieve
        # -------------------------------------------------------

        retrieval = self.retriever.retrieve(question)
        dbg(
            "STEP1_RAG_RETRIEVAL",
            request_id=request_id,
            should_answer=retrieval.should_answer,
            confidence=retrieval.confidence,
            max_similarity=retrieval.max_similarity,
            chunk_count=len(retrieval.chunks),
            chunk_preview=[
                {
                    "similarity": round(chunk.similarity, 4),
                    "text": (chunk.content or "")[:240],
                    "meta": chunk.metadata,
                }
                for chunk in retrieval.chunks[:5]
            ],
        )

        retrieved_docs = document_names(retrieval.chunks)
        qlog(
            "DOCUMENTS",
            path="semantic RAG",
            names=retrieved_docs,
            chunks=len(retrieval.chunks),
            max_similarity=round(retrieval.max_similarity, 4),
            confidence=round(retrieval.confidence, 4),
        )

        # -------------------------------------------------------
        # Step 2 : Similarity Check
        # -------------------------------------------------------

        if not retrieval.should_answer:

            logger.warning(
                "Similarity below threshold. "
                "Skipping LLM."
            )

            total_time = (
                time.perf_counter() - overall_start
            ) * 1000

            qlog(
                "RESULT",
                path="semantic RAG",
                skipped_llm=True,
                reason="below similarity threshold",
                documents=retrieved_docs,
                time_ms=round(total_time, 2),
            )

            answer = (
                "I don't have enough information "
                "in my knowledge base."
            )

            log_payload = {

                "status": "SUCCESS",

                "timestamp": datetime.now().isoformat(),

                "request_id": request_id,

                "provider": self.llm.provider,

                "model": self.llm.model,

                "question": question,

                "answer": answer,

                "confidence": retrieval.confidence,

                "max_similarity":
                    retrieval.max_similarity,

                "should_answer":
                    retrieval.should_answer,

                "chunks_retrieved":
                    len(retrieval.chunks),

                "context_length":
                    len(retrieval.context),

                "retrieval_time_ms":
                    retrieval.retrieval_time_ms,

                "llm_time_ms": 0,

                "llm_provider_latency_ms": 0,

                "total_time_ms":
                    round(total_time, 2),

                "retrieval_threshold":
                    SIMILARITY_THRESHOLD,

                "top_k":
                    TOP_K_RESULTS,

                "temperature":
                    LLM_TEMPERATURE,

                "input_tokens": 0,

                "output_tokens": 0,

                "total_tokens": 0,

                "input_cost": 0,

                "output_cost": 0,

                "embedding_cost": 0,

                "total_cost": 0,

                "sources":
                    retrieval.sources,

                "error": "",
            }

            self.run_logger.log_success(
                log_payload
            )

            return {

                "answer": answer,

                "confidence":
                    retrieval.confidence,

                "provider":
                    self.llm.provider,

                "model":
                    self.llm.model,

                "retrieval_time_ms":
                    retrieval.retrieval_time_ms,

                "llm_time_ms": 0,

                "llm_provider_latency_ms": 0,

                "total_time_ms":
                    round(total_time, 2),

                "chunks_retrieved":
                    len(retrieval.chunks),

                "context_length":
                    len(retrieval.context),

                "retrieval_threshold":
                    SIMILARITY_THRESHOLD,

                "top_k":
                    TOP_K_RESULTS,

                "temperature":
                    LLM_TEMPERATURE,

                "input_tokens": 0,

                "output_tokens": 0,

                "total_tokens": 0,

                "cost": 0,

            }

        # -------------------------------------------------------
        # Step 3 : Prompt
        # -------------------------------------------------------

        messages = self.prompt_builder.build(
            question=question,
            context=retrieval.context,
        )

        # -------------------------------------------------------
        # Step 4 : LLM
        # -------------------------------------------------------

        provider_start = time.perf_counter()

        llm_result = self.llm.generate(
            messages
        )
        dbg(
            "STEP4_LLM_ANSWER",
            request_id=request_id,
            provider=self.llm.provider,
            model=self.llm.model,
            answer=(llm_result.get("answer") or "")[:500],
            input_tokens=llm_result.get("input_tokens"),
            output_tokens=llm_result.get("output_tokens"),
        )

        provider_latency = (
            time.perf_counter() - provider_start
        ) * 1000

        # -------------------------------------------------------
        # Step 5 : Cost
        # -------------------------------------------------------

        cost = self.cost_calculator.calculate(

            provider=self.llm.provider,

            input_tokens=
                llm_result["input_tokens"],

            output_tokens=
                llm_result["output_tokens"],

            embedding_tokens=0,

        )

        # -------------------------------------------------------
        # Step 6 : Total Time
        # -------------------------------------------------------

        total_time = (
            time.perf_counter() - overall_start
        ) * 1000

        qlog(
            "RESULT",
            path="semantic RAG",
            documents=retrieved_docs,
            provider=self.llm.provider,
            model=self.llm.model,
            answer=(llm_result.get("answer") or "")[:240],
            time_ms=round(total_time, 2),
        )

        # -------------------------------------------------------
        # Step 7 : Logging
        # -------------------------------------------------------

        log_payload = {

            "status": "SUCCESS",

            "timestamp":
                datetime.now().isoformat(),

            "request_id":
                request_id,

            "provider":
                self.llm.provider,

            "model":
                self.llm.model,

            "question":
                question,

            "answer":
                llm_result["answer"],

            "confidence":
                retrieval.confidence,

            "max_similarity":
                retrieval.max_similarity,

            "should_answer":
                retrieval.should_answer,

            "chunks_retrieved":
                len(retrieval.chunks),

            "context_length":
                len(retrieval.context),

            "retrieval_time_ms":
                retrieval.retrieval_time_ms,

            "llm_time_ms":
                llm_result["llm_time_ms"],

            "llm_provider_latency_ms":
                round(
                    provider_latency,
                    2,
                ),

            "total_time_ms":
                round(
                    total_time,
                    2,
                ),

            "retrieval_threshold":
                SIMILARITY_THRESHOLD,

            "top_k":
                TOP_K_RESULTS,

            "temperature":
                LLM_TEMPERATURE,

            "input_tokens":
                llm_result["input_tokens"],

            "output_tokens":
                llm_result["output_tokens"],

            "total_tokens":
                llm_result["total_tokens"],

            "input_cost":
                cost["input_cost"],

            "output_cost":
                cost["output_cost"],

            "embedding_cost":
                cost["embedding_cost"],

            "total_cost":
                cost["total_cost"],

            "sources":
                retrieval.sources,

            "error": "",
        }

        self.run_logger.log_success(
            log_payload
        )

        # -------------------------------------------------------
        # Step 8 : Return
        # -------------------------------------------------------

        return {

            "answer":
                llm_result["answer"],

            "confidence":
                retrieval.confidence,

            "provider":
                self.llm.provider,

            "model":
                self.llm.model,

            "sources":
                retrieval.sources,

            "retrieval_time_ms":
                retrieval.retrieval_time_ms,

            "llm_time_ms":
                llm_result["llm_time_ms"],

            "llm_provider_latency_ms":
                round(
                    provider_latency,
                    2,
                ),

            "total_time_ms":
                round(
                    total_time,
                    2,
                ),

            "chunks_retrieved":
                len(retrieval.chunks),

            "context_length":
                len(retrieval.context),

            "retrieval_threshold":
                SIMILARITY_THRESHOLD,

            "top_k":
                TOP_K_RESULTS,

            "temperature":
                LLM_TEMPERATURE,

            "input_tokens":
                llm_result["input_tokens"],

            "output_tokens":
                llm_result["output_tokens"],

            "total_tokens":
                llm_result["total_tokens"],

            "cost":
                cost["total_cost"],

            "cost_breakdown":
                cost,

        }