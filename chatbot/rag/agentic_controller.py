"""
Clarification-driven agentic controller.

Wraps the existing RAG path. Does not replace HybridQA or Chroma.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import time
import uuid

from config import MAX_CLARIFICATION_TURNS, MAX_RETRIEVAL_RETRIES
from logger.console import qlog
from rag.answer_verifier import verify_answer
from rag.calculator import (
    CalculationTrace,
    compute_percentage_change,
    try_explicit_percentage,
)
from rag.clarification import (
    parse_options_from_question,
    period_question,
    should_ask,
    should_ask_period,
)
from rag.conversation_memory import ConversationState, get_memory
from rag.evidence_validator import (
    EvidenceFinding,
    period_options,
    validate_evidence,
)
from rag.query_analyzer import QueryAnalysis, QueryAnalyzer
from rag.query_rewriter import looks_like_clarification, rewrite
from rag.reranker import rerank_chunks


ABSTAIN = "I don't have enough reliable information to answer that."


@dataclass
class AgentPrep:
    conversation_id: str
    original_question: str
    resolved_query: str
    analysis: QueryAnalysis
    memory: ConversationState
    early_response: Optional[Dict[str, Any]] = None
    analysis_time_ms: float = 0.0
    explicit_trace: Optional[CalculationTrace] = None
    clarification_required: bool = False
    user_clarification: str = ""


class AgenticController:
    def __init__(self):
        self.analyzer = QueryAnalyzer()
        self.memory_store = get_memory()

    def prepare(
        self,
        question: str,
        conversation_id: str,
        request_id: str,
    ) -> AgentPrep:
        started = time.perf_counter()
        conversation_id = conversation_id or str(uuid.uuid4())
        memory = self.memory_store.get(conversation_id)
        original = question
        user_clarification = ""

        if looks_like_clarification(question, memory):
            user_clarification = question
            original = memory.original_query or question
            memory.clarification = (
                f"{memory.clarification} {question}".strip()
                if memory.clarification
                else question
            )
            qlog(
                "CLARIFICATION RESOLVED",
                conversation_id=conversation_id,
                user_response=question,
            )

        analysis = self.analyzer.analyze(question, memory)
        if user_clarification:
            merged = self.analyzer.analyze(
                f"{original} {question}",
                memory,
            )
            if not merged.is_ambiguous or merged.metric:
                analysis = merged
                analysis.is_ambiguous = bool(
                    analysis.missing_information
                    and "metric" in analysis.missing_information
                    and not analysis.metric
                )

        resolved = rewrite(question, analysis, memory)
        analysis = self.analyzer.analyze(resolved, memory)

        explicit = try_explicit_percentage(resolved) or try_explicit_percentage(question)

        prep = AgentPrep(
            conversation_id=conversation_id,
            original_question=original,
            resolved_query=resolved,
            analysis=analysis,
            memory=memory,
            analysis_time_ms=(time.perf_counter() - started) * 1000,
            explicit_trace=explicit,
            user_clarification=user_clarification,
        )

        if should_ask(analysis, memory) and not explicit:
            memory.original_query = original
            memory.pending_clarification = True
            memory.pending_slot = (analysis.missing_information or ["metric"])[0]
            memory.clarification_question = analysis.clarification_question
            memory.clarification_options = analysis.clarification_options or (
                parse_options_from_question(analysis.clarification_question)
            )
            memory.clarification_turns += 1
            memory.last_user_message = question
            self.memory_store.save(memory)
            prep.clarification_required = True
            prep.early_response = self._clarification_payload(
                prep,
                request_id,
                analysis.clarification_question,
                memory.clarification_options,
                "ambiguous_metric",
            )
            return prep

        memory.original_query = memory.original_query or original
        memory.resolved_query = resolved
        memory.pending_clarification = False
        memory.last_user_message = question
        if analysis.metric:
            memory.metric = analysis.metric
            memory.last_metric = analysis.metric
        if analysis.start_period:
            memory.start_period = analysis.start_period
            memory.last_start_period = analysis.start_period
        if analysis.end_period:
            memory.end_period = analysis.end_period
            memory.last_end_period = analysis.end_period
        self.memory_store.save(memory)
        qlog(
            "QUERY TYPE",
            intent=analysis.intent,
            resolved=resolved,
            metric=analysis.metric,
            ambiguous=analysis.is_ambiguous,
        )
        return prep

    def after_retrieval(
        self,
        prep: AgentPrep,
        chunks: List,
        request_id: str,
        retrieval_attempts: int,
    ) -> Dict[str, Any]:
        rerank_started = time.perf_counter()
        ranked = rerank_chunks(prep.resolved_query, chunks, prep.analysis)
        rerank_ms = (time.perf_counter() - rerank_started) * 1000
        finding = validate_evidence(ranked, prep.analysis)

        result = {
            "chunks": ranked,
            "finding": finding,
            "rerank_ms": rerank_ms,
            "early_response": None,
            "trace": prep.explicit_trace,
            "retrieval_attempts": retrieval_attempts,
        }

        if (
            should_ask_period(prep.analysis, prep.memory, period_options(finding.periods))
            and not prep.explicit_trace
        ):
            options = period_options(finding.periods)
            question = period_question(options)
            prep.memory.original_query = prep.original_question
            prep.memory.pending_clarification = True
            prep.memory.pending_slot = "period"
            prep.memory.clarification_question = question
            prep.memory.clarification_options = options
            prep.memory.clarification_turns += 1
            prep.memory.metric = prep.analysis.metric or prep.memory.metric
            self.memory_store.save(prep.memory)
            result["early_response"] = self._clarification_payload(
                prep,
                request_id,
                question,
                options,
                "missing_period",
            )
            return result

        if prep.analysis.requires_calculation and not prep.explicit_trace:
            trace = self._trace_from_finding(prep.analysis, finding)
            result["trace"] = trace
            if finding.status == "conflicting":
                result["early_response"] = self._status_payload(
                    prep,
                    request_id,
                    finding.message,
                    "CONFLICTING_EVIDENCE",
                    finding,
                    0.2,
                )
            elif finding.status in {"insufficient", "irrelevant"}:
                if retrieval_attempts < MAX_RETRIEVAL_RETRIES and finding.status == "insufficient":
                    result["retry"] = True
                else:
                    result["early_response"] = self._status_payload(
                        prep,
                        request_id,
                        finding.message or ABSTAIN,
                        "INSUFFICIENT_EVIDENCE",
                        finding,
                        finding.completeness * 0.4,
                    )
        elif finding.status in {"insufficient", "irrelevant", "conflicting"}:
            status = (
                "CONFLICTING_EVIDENCE"
                if finding.status == "conflicting"
                else "INSUFFICIENT_EVIDENCE"
            )
            result["early_response"] = self._status_payload(
                prep,
                request_id,
                finding.message or ABSTAIN,
                status,
                finding,
                finding.completeness * 0.4,
            )
        return result

    def calculation_note(self, trace: Optional[CalculationTrace]) -> str:
        if not trace or not trace.verified:
            return ""
        return (
            f"formula: {trace.formula}\n"
            f"inputs: {trace.inputs}\n"
            f"result: {trace.result}\n"
        )

    def finalize_answer(
        self,
        answer: str,
        evidence_text: str,
        trace: Optional[CalculationTrace],
        finding: Optional[EvidenceFinding],
        retrieval_confidence: float,
        answer_verified: Optional[dict] = None,
    ) -> Dict[str, Any]:
        check = answer_verified or verify_answer(answer, evidence_text, trace)
        if not check["verified"]:
            return {
                "answer": ABSTAIN,
                "answer_verified": False,
                "verification_reason": check["reason"],
                "confidence": 0.15,
                "status": "VERIFICATION_FAILED",
            }

        completeness = finding.completeness if finding else 1.0
        calc_ok = 1.0 if (trace is None or trace.verified) else 0.0
        confidence = (
            (0.35 * float(retrieval_confidence or 0.0))
            + (0.35 * float(completeness))
            + (0.15 * calc_ok)
            + 0.15
        )
        return {
            "answer": answer,
            "answer_verified": True,
            "verification_reason": check["reason"],
            "confidence": round(min(1.0, confidence), 4),
            "status": "SUCCESS",
        }

    def remember_answer(self, prep: AgentPrep, answer: str) -> None:
        prep.memory.last_answer = answer
        prep.memory.resolved_query = prep.resolved_query
        prep.memory.pending_clarification = False
        self.memory_store.save(prep.memory)

    def _trace_from_finding(
        self,
        analysis: QueryAnalysis,
        finding: EvidenceFinding,
    ) -> Optional[CalculationTrace]:
        if not analysis.metric or not analysis.start_period or not analysis.end_period:
            return None
        old_key = f"{analysis.start_period}_{analysis.metric}"
        new_key = f"{analysis.end_period}_{analysis.metric}"
        if old_key not in finding.found_values or new_key not in finding.found_values:
            return None
        try:
            return compute_percentage_change(
                finding.found_values[old_key],
                finding.found_values[new_key],
            )
        except Exception:
            return None

    def _clarification_payload(
        self,
        prep: AgentPrep,
        request_id: str,
        question: str,
        options: List[str],
        reason: str,
    ) -> Dict[str, Any]:
        qlog(
            "CLARIFICATION",
            reason=reason,
            question=question,
            turns=prep.memory.clarification_turns,
        )
        return self._base_result(
            prep,
            request_id,
            answer=question,
            status="CLARIFICATION_REQUIRED",
            confidence=0.0,
            extra={
                "clarification_required": True,
                "clarification_question": question,
                "clarification_options": options,
                "clarification_reason": reason,
            },
        )

    def _status_payload(
        self,
        prep: AgentPrep,
        request_id: str,
        answer: str,
        status: str,
        finding: EvidenceFinding,
        confidence: float,
    ) -> Dict[str, Any]:
        qlog("RESULT", path="agentic", status=status, answer=answer[:240])
        return self._base_result(
            prep,
            request_id,
            answer=answer,
            status=status,
            confidence=round(confidence, 4),
            extra={
                "evidence_completeness": finding.completeness,
                "extracted_values": finding.found_values,
            },
        )

    def _base_result(
        self,
        prep: AgentPrep,
        request_id: str,
        answer: str,
        status: str,
        confidence: float,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        payload = {
            "answer": answer,
            "confidence": confidence,
            "provider": "agentic",
            "model": "clarification-controller",
            "sources": [],
            "retrieval_time_ms": 0,
            "llm_time_ms": 0,
            "llm_provider_latency_ms": 0,
            "total_time_ms": 0,
            "chunks_retrieved": 0,
            "context_length": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "cost": 0,
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
        if extra:
            payload.update(extra)
        return payload
