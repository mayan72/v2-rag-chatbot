from rag.calculator import compute_percentage_change, try_explicit_percentage
from rag.conversation_memory import ConversationMemory, ConversationState
from rag.evidence_validator import validate_evidence
from rag.query_analyzer import QueryAnalysis, QueryAnalyzer
from rag.query_rewriter import apply_clarification, rewrite
from rag.answer_verifier import verify_answer
from rag.clarification import should_ask


class Chunk:
    def __init__(self, content, similarity=0.8):
        self.content = content
        self.similarity = similarity


def test_ambiguous_growth_asks_for_metric():
    analysis = QueryAnalyzer().analyze("What was the growth?")
    memory = ConversationState(conversation_id="t1")
    assert analysis.is_ambiguous is True
    assert "metric" in analysis.missing_information
    assert should_ask(analysis, memory) is True
    assert "revenue growth" in analysis.clarification_question.lower()


def test_ambiguous_percentage():
    analysis = QueryAnalyzer().analyze("What was the percentage?")
    assert analysis.is_ambiguous is True


def test_revenue_growth_needs_period():
    analysis = QueryAnalyzer().analyze("What was the revenue growth?")
    assert analysis.metric == "revenue"
    assert analysis.needs_period is True
    assert analysis.requires_calculation is True


def test_rewrite_after_metric_clarification():
    analysis = QueryAnalyzer().analyze("Revenue growth.")
    analysis.metric = "revenue"
    resolved = apply_clarification(
        "What was the growth?",
        "Revenue growth.",
        analysis,
    )
    assert "revenue growth" in resolved.lower()


def test_follow_up_growth_uses_memory():
    memory = ConversationState(
        conversation_id="t2",
        last_metric="revenue",
        last_start_period="2023",
        last_end_period="2024",
    )
    analysis = QueryAnalyzer().analyze("And growth in 2024?", memory)
    resolved = rewrite("And growth in 2024?", analysis, memory)
    assert "revenue" in resolved.lower()
    assert "2023" in resolved
    assert "2024" in resolved


def test_explicit_percentage_uses_calculator():
    trace = try_explicit_percentage(
        "What is the percentage increase from 500 to 600?"
    )
    assert trace is not None
    assert trace.verified is True
    assert abs(trace.result - 20.0) < 1e-6


def test_missing_2024_value_is_insufficient():
    analysis = QueryAnalysis(
        intent="numerical",
        requires_calculation=True,
        metric="revenue",
        start_period="2023",
        end_period="2024",
        required_values=["2023 revenue", "2024 revenue"],
    )
    chunks = [
        Chunk("2023 Revenue = 500 million"),
    ]
    finding = validate_evidence(chunks, analysis)
    assert finding.status == "insufficient"
    assert "2024" in finding.message


def test_both_years_enable_growth_calc():
    analysis = QueryAnalysis(
        intent="numerical",
        requires_calculation=True,
        metric="revenue",
        start_period="2023",
        end_period="2024",
    )
    chunks = [
        Chunk("2023 Revenue = 500\n2024 Revenue = 600"),
    ]
    finding = validate_evidence(chunks, analysis)
    assert finding.status == "relevant"
    trace = compute_percentage_change(
        finding.found_values["2023_revenue"],
        finding.found_values["2024_revenue"],
    )
    assert abs(trace.result - 20.0) < 1e-6


def test_missing_evidence_milk():
    analysis = QueryAnalysis(
        intent="factual",
        metric="milk",
        start_period="2024",
    )
    chunks = [
        Chunk("The company reported aluminium production in 2024."),
    ]
    finding = validate_evidence(chunks, analysis)
    assert finding.status == "insufficient"


def test_conflicting_evidence():
    analysis = QueryAnalysis(
        intent="numerical",
        requires_calculation=True,
        metric="revenue",
        start_period="2023",
        end_period="2024",
    )
    chunks = [
        Chunk("2023 revenue was 500. 2023 revenue was 800. 2024 revenue was 600."),
    ]
    finding = validate_evidence(chunks, analysis)
    assert finding.status == "conflicting"


def test_answer_verifier_rejects_invented_number():
    check = verify_answer(
        "Revenue was 99999 in 2024.",
        "2024 Revenue = 600",
        None,
    )
    assert check["verified"] is False


def test_memory_roundtrip():
    store = ConversationMemory()
    state = store.get("abc")
    state.metric = "revenue"
    store.save(state)
    assert store.get("abc").metric == "revenue"
