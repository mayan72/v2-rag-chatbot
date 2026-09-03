"""
Validate whether retrieved text actually contains required facts.

Does not trust Chroma similarity alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import re

from rag.query_analyzer import QueryAnalysis


NUMBER_RE = re.compile(
    r"(?<![\w])(-?\d{1,3}(?:,\d{3})*(?:\.\d+)?|-?\d+(?:\.\d+)?)(?![\w])"
)
YEAR_RE = re.compile(r"\b((?:19|20)\d{2})\b")


@dataclass
class EvidenceFinding:
    status: str
    found_values: Dict[str, float] = field(default_factory=dict)
    periods: List[str] = field(default_factory=list)
    conflicts: List[str] = field(default_factory=list)
    completeness: float = 0.0
    message: str = ""

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "found_values": self.found_values,
            "periods": self.periods,
            "conflicts": self.conflicts,
            "completeness": self.completeness,
            "message": self.message,
        }


def _parse_number(raw: str) -> Optional[float]:
    try:
        return float(raw.replace(",", ""))
    except (TypeError, ValueError):
        return None


def extract_years(text: str) -> List[str]:
    years = YEAR_RE.findall(text or "")
    unique = []
    for year in years:
        if year not in unique:
            unique.append(year)
    return unique


def extract_metric_values(text: str, metric: str) -> Dict[str, List[float]]:
    """Map year -> values mentioned near the metric."""
    if not text or not metric:
        return {}

    lowered = text.lower()
    metric_l = metric.lower()
    if metric_l not in lowered and not any(
        alias in lowered for alias in (metric_l,)
    ):
        # still allow year-number pairs when the question already named the metric
        pass

    values: Dict[str, List[float]] = {}
    lines = re.split(r"[\n.;]", text)
    for line in lines:
        years = extract_years(line)
        numbers = [_parse_number(item) for item in NUMBER_RE.findall(line)]
        numbers = [item for item in numbers if item is not None]
        numbers = [
            item for item in numbers
            if item not in {float(year) for year in years}
        ]
        if not numbers:
            continue
        line_l = line.lower()
        metric_here = metric_l in line_l
        if years:
            for year in years:
                if metric_here or metric_l in lowered:
                    values.setdefault(year, []).extend(numbers[:2])
        elif metric_here:
            values.setdefault("unknown", []).extend(numbers[:2])
    return values


def period_options(years: List[str]) -> List[str]:
    ordered = sorted(set(years))
    options = []
    for index in range(len(ordered) - 1):
        options.append(f"{ordered[index]}-{ordered[index + 1]}")
    return options


def validate_evidence(
    chunks: List,
    analysis: QueryAnalysis,
) -> EvidenceFinding:
    combined = "\n".join(
        getattr(chunk, "content", "") or ""
        for chunk in chunks
    )
    if not combined.strip():
        return EvidenceFinding(
            status="irrelevant",
            completeness=0.0,
            message=(
                "I couldn't find information in the provided data, "
                "so I can't reliably calculate or report it."
            ),
        )

    years = extract_years(combined)
    finding = EvidenceFinding(periods=years, status="relevant")

    if not analysis.requires_calculation:
        finding.completeness = 1.0 if chunks else 0.0
        if analysis.metric and analysis.metric.lower() not in combined.lower():
            finding.status = "insufficient"
            finding.completeness = 0.2
            finding.message = (
                f"I couldn't find information about {analysis.metric} "
                "in the provided data, so I can't reliably calculate or report it."
            )
        return finding

    metric = analysis.metric
    if metric:
        extracted = extract_metric_values(combined, metric)
        for year, numbers in extracted.items():
            unique = []
            for number in numbers:
                if unique and any(abs(number - existing) > 1e-6 for existing in unique):
                    finding.conflicts.append(
                        f"Conflicting {metric} values for {year}: {unique + [number]}"
                    )
                if number not in unique:
                    unique.append(number)
            if unique:
                key = f"{year}_{metric}" if year != "unknown" else metric
                finding.found_values[key] = unique[0]
    else:
        extracted = {}

    if finding.conflicts:
        finding.status = "conflicting"
        finding.completeness = 0.4
        finding.message = (
            "The retrieved sources contain conflicting values for the same metric, "
            "so I can't reliably report a single number."
        )
        return finding

    required_years = [
        year for year in (analysis.start_period, analysis.end_period) if year
    ]
    if required_years and metric:
        missing = []
        for year in required_years:
            key = f"{year}_{metric}"
            if key not in finding.found_values:
                missing.append(year)
        found = len(required_years) - len(missing)
        finding.completeness = found / len(required_years)
        if missing:
            finding.status = "insufficient"
            if found:
                found_years = [year for year in required_years if year not in missing]
                finding.message = (
                    f"I found the {found_years[0]} {metric}, but I couldn't find "
                    f"the {missing[0]} {metric} in the available data. "
                    "I can't reliably calculate the growth."
                )
            else:
                finding.message = (
                    f"I couldn't find information about {metric} "
                    f"in {', '.join(required_years)} in the provided data, "
                    "so I can't reliably calculate or report it."
                )
            return finding
        finding.status = "relevant"
        return finding

    if analysis.needs_period:
        finding.completeness = 0.5 if years else 0.0
        return finding

    if metric and metric.lower() not in combined.lower():
        finding.status = "insufficient"
        finding.completeness = 0.0
        finding.message = (
            f"I couldn't find information about {metric} "
            "in the provided data, so I can't reliably calculate or report it."
        )
        return finding

    finding.completeness = 1.0
    return finding
