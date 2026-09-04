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
from typing import Dict, Any, Optional
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
from rag.agentic_controller import AgenticController
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
        self.agent = AgenticController()

        self.cost_calculator = CostCalculator()

        qlog("CHATBOT", status="ready")

        # ============================================================
    # Private Methods
    # ============================================================

    
        # ============================================================

    def ask(
        self,
        question: str,
        conversation_id: str = "",
    ) -> Dict[str, Any]:
        """
        Execute the complete RAG pipeline.
        """

        overall_start = time.perf_counter()

        request_id = str(uuid.uuid4())
        qlog("QUESTION", text=question)
        dbg("ASK_START", request_id=request_id, question=question)

        prep = self.agent.prepare(
            question=question,
            conversation_id=conversation_id,
            request_id=request_id,
        )
        if prep.early_response:
            total_time = (time.perf_counter() - overall_start) * 1000
            prep.early_response["total_time_ms"] = round(total_time, 2)
            self._log_agent_event(prep, prep.early_response, request_id)
            return prep.early_response

        lookup_question = prep.resolved_query or question

        if prep.explicit_trace and prep.explicit_trace.verified:
            total_time = (time.perf_counter() - overall_start) * 1000
            trace = prep.explicit_trace
            answer = (
                f"The value increased from {trace.inputs['old_value']} to "
                f"{trace.inputs['new_value']}, representing a "
                f"{round(trace.result, 2)}% change."
            )
            if trace.inputs["new_value"] < trace.inputs["old_value"]:
                answer = (
                    f"The value decreased from {trace.inputs['old_value']} to "
                    f"{trace.inputs['new_value']}, representing a "
                    f"{round(trace.result, 2)}% change."
                )
            result = self.agent._base_result(
                prep,
                request_id,
                answer=answer,
                status="SUCCESS",
                confidence=1.0,
                extra={
                    "formula": trace.formula,
                    "calculation_result": trace.result,
                    "calculation_verified": True,
                    "numerical_query": True,
                    "provider": "calculator",
                    "model": "deterministic",
                    "total_time_ms": round(total_time, 2),
                },
            )
            self.agent.remember_answer(prep, answer)
            self._log_agent_event(prep, result, request_id)
            return result

        # -------------------------------------------------------
        # Step 0 : Structured table QA (counts / sums / filters)
        # -------------------------------------------------------

        structured = self.hybrid_qa.answer(lookup_question)
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
            self.agent.remember_answer(prep, structured.answer)
            return self._with_agent_fields(
                {
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
                },
                prep,
                request_id,
                status="SUCCESS",
            )

        # -------------------------------------------------------
        # Step 0 : Structured table QA (counts / sums / filters)
        # -------------------------------------------------------

        structured = self.hybrid_qa.answer(lookup_question)

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
            self.agent.remember_answer(prep, structured.answer)
            return self._with_agent_fields(
                {
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
                },
                prep,
                request_id,
                status="SUCCESS",
            )

        if structured and getattr(structured, "blocked", False):
            total_time = (
                time.perf_counter() - overall_start
            ) * 1000
            answer = (
                structured.answer
                or structured.error
                or (
                    "I could not apply the requested filter to the "
                    "table, so I will not return an unfiltered total."
                )
            )
            qlog(
                "RESULT",
                path="structured-blocked",
                documents=getattr(structured, "document_name", ""),
                operation=structured.operation,
                filters=format_filters(structured.filters),
                answer=answer[:240],
                time_ms=round(total_time, 2),
            )
            self.run_logger.log_failure(
                question=question,
                error=RuntimeError(answer),
                stage="structured_filter",
            )
            return self._with_agent_fields(
                {
                    "answer": answer,
                    "confidence": 0.15,
                    "provider": "structured",
                    "model": "table-engine",
                    "sources": [],
                    "retrieval_time_ms": 0,
                    "llm_time_ms": 0,
                    "llm_provider_latency_ms": 0,
                    "total_time_ms": round(total_time, 2),
                    "chunks_retrieved": 0,
                    "context_length": 0,
                    "retrieval_threshold": SIMILARITY_THRESHOLD,
                    "top_k": TOP_K_RESULTS,
                    "temperature": LLM_TEMPERATURE,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "cost": 0,
                    "should_answer": False,
                },
                prep,
                request_id,
                status="BLOCKED",
            )

        # -------------------------------------------------------
        # Step 1 : Retrieve
        # -------------------------------------------------------

        retrieval = self.retriever.retrieve(lookup_question)
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

        post = self.agent.after_retrieval(
            prep=prep,
            chunks=retrieval.chunks,
            request_id=request_id,
            retrieval_attempts=1,
        )
        if post.get("retry"):
            retry_query = " ".join(
                part for part in (
                    prep.analysis.metric,
                    prep.analysis.start_period,
                    prep.analysis.end_period,
                    lookup_question,
                ) if part
            )
            retrieval = self.retriever.retrieve(retry_query)
            post = self.agent.after_retrieval(
                prep=prep,
                chunks=retrieval.chunks,
                request_id=request_id,
                retrieval_attempts=2,
            )
        if post.get("chunks"):
            retrieval.chunks = post["chunks"]
        if post.get("early_response"):
            early = post["early_response"]
            early["sources"] = retrieval.sources or []
            early["retrieval_time_ms"] = retrieval.retrieval_time_ms
            early["chunks_retrieved"] = len(retrieval.chunks)
            early["total_time_ms"] = round(
                (time.perf_counter() - overall_start) * 1000, 2
            )
            self._log_agent_event(prep, early, request_id)
            return early

        calc_trace = post.get("trace")
        finding = post.get("finding")

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

            return self._with_agent_fields(
                {

                "answer": answer,

                "confidence":
                    retrieval.confidence,

                "provider":
                    self.llm.provider,

                "model":
                    self.llm.model,

                "sources":
                    retrieval.sources or [],

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

                },
                prep,
                request_id,
                status="INSUFFICIENT_EVIDENCE",
            )

        # -------------------------------------------------------
        # Step 3 : Prompt
        # -------------------------------------------------------

        messages = self.prompt_builder.build(
            question=lookup_question,
            context=retrieval.context,
            calculation_note=self.agent.calculation_note(calc_trace),
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

        evidence_text = retrieval.context or ""
        finalized = self.agent.finalize_answer(
            answer=llm_result["answer"],
            evidence_text=evidence_text,
            trace=calc_trace,
            finding=finding,
            retrieval_confidence=retrieval.confidence,
        )
        final_answer = finalized["answer"]
        final_confidence = finalized["confidence"]
        final_status = finalized["status"]
        self.agent.remember_answer(prep, final_answer)

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
                lookup_question,

            "original_question":
                prep.original_question,

            "resolved_question":
                prep.resolved_query,

            "answer":
                final_answer,

            "confidence":
                final_confidence,

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
            "status": final_status,
            "conversation_id": prep.conversation_id,
            "intent": prep.analysis.intent,
            "formula": getattr(calc_trace, "formula", None),
            "calculation_result": getattr(calc_trace, "result", None),
            "calculation_verified": getattr(calc_trace, "verified", None),
            "answer_verified": finalized.get("answer_verified"),
            "evidence_completeness": getattr(finding, "completeness", None),
        }

        self.run_logger.log_success(
            log_payload
        )

        # -------------------------------------------------------
        # Step 8 : Return
        # -------------------------------------------------------

        return self._with_agent_fields(
            {

            "answer":
                final_answer,

            "confidence":
                final_confidence,

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

            },
            prep,
            request_id,
            status=final_status,
            extra={
                "formula": getattr(calc_trace, "formula", None),
                "calculation_result": getattr(calc_trace, "result", None),
                "calculation_verified": getattr(calc_trace, "verified", None),
                "answer_verified": finalized.get("answer_verified"),
                "evidence_completeness": getattr(finding, "completeness", None),
                "reranking_used": True,
            },
        )

    def _with_agent_fields(
        self,
        result: Dict[str, Any],
        prep,
        request_id: str,
        status: str = "SUCCESS",
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        result.setdefault("sources", [])
        result.update(
            {
                "request_id": request_id,
                "conversation_id": prep.conversation_id,
                "original_question": prep.original_question,
                "resolved_question": prep.resolved_query,
                "intent": prep.analysis.intent,
                "is_ambiguous": prep.analysis.is_ambiguous,
                "clarification_required": False,
                "clarification_question": None,
                "clarification_options": [],
                "status": status,
                "numerical_query": prep.analysis.numerical,
                "calculation_required": prep.analysis.requires_calculation,
            }
        )
        if extra:
            result.update(extra)
        return result

    def _log_agent_event(self, prep, result: Dict[str, Any], request_id: str) -> None:
        payload = {
            "status": result.get("status", "SUCCESS"),
            "timestamp": datetime.now().isoformat(),
            "request_id": request_id,
            "conversation_id": prep.conversation_id,
            "provider": result.get("provider", ""),
            "model": result.get("model", ""),
            "question": prep.original_question,
            "original_question": prep.original_question,
            "resolved_question": prep.resolved_query,
            "answer": result.get("answer", ""),
            "confidence": result.get("confidence", 0),
            "max_similarity": 0,
            "should_answer": not result.get("clarification_required"),
            "chunks_retrieved": result.get("chunks_retrieved", 0),
            "context_length": result.get("context_length", 0),
            "retrieval_time_ms": result.get("retrieval_time_ms", 0),
            "llm_time_ms": result.get("llm_time_ms", 0),
            "llm_provider_latency_ms": result.get("llm_provider_latency_ms", 0),
            "total_time_ms": result.get("total_time_ms", 0),
            "retrieval_threshold": SIMILARITY_THRESHOLD,
            "top_k": TOP_K_RESULTS,
            "temperature": LLM_TEMPERATURE,
            "input_tokens": result.get("input_tokens", 0),
            "output_tokens": result.get("output_tokens", 0),
            "total_tokens": result.get("total_tokens", 0),
            "input_cost": 0,
            "output_cost": 0,
            "embedding_cost": 0,
            "total_cost": result.get("cost", 0),
            "sources": result.get("sources", []),
            "error": "",
            "intent": prep.analysis.intent,
            "clarification_required": result.get("clarification_required", False),
            "clarification_question": result.get("clarification_question"),
        }
        self.run_logger.log_success(payload)
