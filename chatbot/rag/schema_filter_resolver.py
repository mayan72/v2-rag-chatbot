"""
Schema-driven filter resolution for structured aggregations.

Uses only the uploaded table's column names and values.
Does not hardcode business terms (furniture, category, region, etc.).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from rag.text_normalize import best_value_match, normalize_text


IsNumeric = Callable[[dict], bool]

OPERATION_STOPWORDS = {
    "sum", "total", "count", "average", "avg", "mean", "min", "max",
    "median", "std", "variance", "correlation", "what", "whats", "is",
    "the", "of", "a", "an", "give", "me", "please", "calculate", "find",
    "show", "how", "many", "much", "and", "or", "for", "in", "on", "to",
    "from", "by", "with", "as", "it", "this", "that", "all", "rows",
    "row", "value", "values", "number", "amount", "where", "equals",
    "equal", "get", "tell", "greater", "least", "most", "above", "below",
    "under", "than", "between", "over", "at", "least", "combined",
    "add", "up", "calculate", "compute", "report", "list",
    "sold", "sale", "generated", "earned", "booked", "recorded",
    "reported", "achieved", "data", "table", "sheet", "file",
    "excel", "csv", "xlsx",
}

WEAK_COLUMN_ALIASES = {
    "id", "ids", "code", "key", "name", "names", "value", "values",
    "date", "time", "no", "number",
}


@dataclass
class FilterResolution:
    filters: list
    leftover: List[str]

    @property
    def unresolved(self) -> bool:
        return bool(self.leftover)


class SchemaFilterResolver:
    """Resolve equality filters from a question against an arbitrary schema."""

    def __init__(
        self,
        column_min_score: float = 0.78,
        value_min_score: float = 0.88,
    ):
        self.column_min_score = column_min_score
        self.value_min_score = value_min_score

    def resolve(
        self,
        question: str,
        columns: Sequence[dict],
        extra_values: Optional[Dict[str, Iterable[Any]]] = None,
        is_numeric: Optional[IsNumeric] = None,
    ) -> FilterResolution:
        extra_values = extra_values or {}
        dim_columns = self._dimension_columns(columns, is_numeric)
        aliases = self._column_aliases(dim_columns)
        value_index = self._value_index(dim_columns, extra_values)

        filters = []
        consumed: set[str] = set()

        filters.extend(
            self._extract_column_bound_phrases(
                question=question,
                aliases=aliases,
                value_index=value_index,
                consumed=consumed,
            )
        )
        filters.extend(
            self._extract_preposition_values(
                question=question,
                dim_columns=dim_columns,
                value_index=value_index,
                used={item.column for item in filters},
                consumed=consumed,
            )
        )
        filters.extend(
            self._extract_unique_value_mentions(
                question=question,
                dim_columns=dim_columns,
                value_index=value_index,
                used={item.column for item in filters},
                consumed=consumed,
            )
        )

        filters = self._dedupe(filters)
        leftover = self.leftover_tokens(
            question=question,
            columns=columns,
            filters=filters,
            extra_consumed=consumed,
        )
        return FilterResolution(filters=filters, leftover=leftover)

    def leftover_tokens(
        self,
        question: str,
        columns: Sequence[dict],
        filters: Sequence[Any],
        extra_consumed: Optional[set] = None,
    ) -> List[str]:
        names = set()
        for column in columns:
            names.update(
                self._name_tokens(str(column.get("name") or ""))
            )

        consumed = set(extra_consumed or ())
        for item in filters:
            consumed.update(normalize_text(item.value).split())
            requested = getattr(item, "requested_value", None)
            if requested is not None:
                consumed.update(normalize_text(requested).split())
            consumed.update(self._name_tokens(str(item.column)))

        leftover = []
        for token in normalize_text(question).split():
            if token in OPERATION_STOPWORDS:
                continue
            if self._is_column_token(token, names):
                continue
            if token in consumed:
                continue
            if len(token) < 3:
                continue
            if token.isdigit():
                continue
            leftover.append(token)
        return leftover

    def _dimension_columns(
        self,
        columns: Sequence[dict],
        is_numeric: Optional[IsNumeric],
    ) -> List[dict]:
        result = []
        for column in columns:
            if column.get("internal"):
                continue
            if is_numeric and is_numeric(column):
                continue
            name = str(column.get("name") or "").strip()
            if not name:
                continue
            result.append(column)
        return result

    def _name_tokens(self, name: str) -> set:
        normalized = normalize_text(name)
        tokens = set(normalized.split()) if normalized else set()
        if normalized:
            tokens.add(normalized)
        return tokens

    def _is_column_token(self, token: str, names: set) -> bool:
        if token in names:
            return True
        if token.endswith("s") and token[:-1] in names:
            return True
        if token.endswith("ies") and (token[:-3] + "y") in names:
            return True
        return False

    def _column_aliases(
        self,
        dim_columns: Sequence[dict],
    ) -> List[Tuple[str, str]]:
        token_owners: Dict[str, List[str]] = {}
        aliases: List[Tuple[str, str]] = []
        for column in dim_columns:
            name = str(column.get("name") or "")
            normalized = normalize_text(name)
            if not normalized:
                continue
            aliases.append((normalized, name))
            tokens = [token for token in normalized.split() if len(token) >= 3]
            for token in tokens:
                token_owners.setdefault(token, []).append(name)

        for token, owners in token_owners.items():
            if token in WEAK_COLUMN_ALIASES:
                continue
            unique_owners = list(dict.fromkeys(owners))
            if len(unique_owners) == 1:
                aliases.append((token, unique_owners[0]))

        aliases.sort(key=lambda item: len(item[0]), reverse=True)
        return aliases

    def _values_for(
        self,
        column: dict,
        extra_values: Dict[str, Iterable[Any]],
    ) -> List[Any]:
        name = str(column.get("name") or "")
        values = list(column.get("sample_values") or [])
        extra = extra_values.get(name) or []
        seen = {normalize_text(item) for item in values if normalize_text(item)}
        for item in extra:
            key = normalize_text(item)
            if not key or key in seen:
                continue
            seen.add(key)
            values.append(item)
        return values

    def _value_index(
        self,
        dim_columns: Sequence[dict],
        extra_values: Dict[str, Iterable[Any]],
    ) -> Dict[str, List[Any]]:
        return {
            str(column.get("name")): self._values_for(column, extra_values)
            for column in dim_columns
        }

    def _match_value(
        self,
        requested: str,
        values: Iterable[Any],
    ) -> Optional[Tuple[Any, float]]:
        requested = (requested or "").strip()
        if not requested:
            return None
        if normalize_text(requested) in OPERATION_STOPWORDS:
            return None
        if not list(values):
            return None
        return best_value_match(
            requested,
            values,
            min_score=self.value_min_score,
        )

    def _extract_column_bound_phrases(
        self,
        question: str,
        aliases: List[Tuple[str, str]],
        value_index: Dict[str, List[Any]],
        consumed: set,
    ) -> List:
        from rag.query_planner import QueryFilter

        text = normalize_text(question)
        filters = []
        used_columns = set()

        for alias, column_name in aliases:
            if column_name in used_columns:
                continue
            alias_re = re.escape(alias)
            patterns = [
                rf"\bof\s+(?:the\s+)?(?P<value>.+?)\s+{alias_re}\b",
                rf"\b{alias_re}\s+(?:of|is|=|equals?)\s+(?:the\s+)?(?P<value>.+?)(?:\s+(?:for|in|and|with)\b|$)",
                rf"\bwhere\s+{alias_re}\s+(?:is|=|equals?)\s+(?:the\s+)?(?P<value>.+?)(?:\s+(?:for|in|and|with)\b|$)",
                rf"\b(?P<value>.+?)\s+{alias_re}\b",
                rf"\b{alias_re}\s+(?P<value>.+?)(?:\s+(?:for|in|and|with)\b|$)",
            ]
            for pattern in patterns:
                match = re.search(pattern, text)
                if not match:
                    continue
                value_text = match.group("value").strip()
                value_text = re.split(
                    r"\b(?:for|in|and|with|of|where)\b",
                    value_text,
                    maxsplit=1,
                )[0].strip()
                if not value_text or value_text == alias:
                    continue
                if normalize_text(value_text) in OPERATION_STOPWORDS:
                    continue
                matched = self._match_value(
                    value_text,
                    value_index.get(column_name) or [],
                )
                if matched is None:
                    continue
                value, score = matched
                filters.append(
                    QueryFilter(
                        column=column_name,
                        op="eq",
                        value=value,
                        score=score,
                        validated=score >= self.value_min_score,
                        requested_value=value_text,
                    )
                )
                used_columns.add(column_name)
                consumed.update(normalize_text(value).split())
                consumed.update(normalize_text(value_text).split())
                consumed.update(alias.split())
                break
        return filters

    def _extract_preposition_values(
        self,
        question: str,
        dim_columns: Sequence[dict],
        value_index: Dict[str, List[Any]],
        used: set,
        consumed: set,
    ) -> List:
        from rag.query_planner import QueryFilter

        text = normalize_text(question)
        filters = []
        for pattern in (
            r"\bfor\s+(?:the\s+)?(.+)$",
            r"\bin\s+(?:the\s+)?(.+)$",
        ):
            match = re.search(pattern, text)
            if not match:
                continue
            value_text = match.group(1).strip()
            value_text = re.split(
                r"\b(?:and|with)\b",
                value_text,
                maxsplit=1,
            )[0].strip()
            # Drop a leading column alias so "for ship mode first class"
            # still matches the value "first class".
            stripped = self._strip_leading_column_alias(
                value_text,
                dim_columns,
            )
            hit = self._unique_column_value(
                stripped,
                dim_columns,
                value_index,
                used,
            )
            if hit is None and stripped != value_text:
                hit = self._unique_column_value(
                    value_text,
                    dim_columns,
                    value_index,
                    used,
                )
            if hit is None:
                continue
            column_name, value, score = hit
            filters.append(
                QueryFilter(
                    column=column_name,
                    op="eq",
                    value=value,
                    score=score,
                    validated=score >= self.value_min_score,
                    requested_value=value_text,
                )
            )
            used.add(column_name)
            consumed.update(normalize_text(value).split())
            consumed.update(normalize_text(value_text).split())
        return filters

    def _strip_leading_column_alias(
        self,
        value_text: str,
        dim_columns: Sequence[dict],
    ) -> str:
        text = normalize_text(value_text)
        aliases = sorted(
            (
                normalize_text(column.get("name") or "")
                for column in dim_columns
            ),
            key=len,
            reverse=True,
        )
        for alias in aliases:
            if alias and text.startswith(alias + " "):
                return text[len(alias):].strip()
        return value_text

    def _extract_unique_value_mentions(
        self,
        question: str,
        dim_columns: Sequence[dict],
        value_index: Dict[str, List[Any]],
        used: set,
        consumed: set,
    ) -> List:
        from rag.query_planner import QueryFilter

        tokens = [
            token
            for token in normalize_text(question).split()
            if token not in OPERATION_STOPWORDS
        ]
        column_tokens = set()
        for column in dim_columns:
            column_tokens.update(
                self._name_tokens(str(column.get("name") or ""))
            )

        ngrams = []
        for size in (4, 3, 2, 1):
            for index in range(0, max(0, len(tokens) - size + 1)):
                ngrams.append(" ".join(tokens[index:index + size]))

        filters = []
        for ngram in ngrams:
            if not ngram or ngram in column_tokens:
                continue
            if all(token in consumed for token in ngram.split()):
                continue
            hit = self._unique_column_value(
                ngram,
                dim_columns,
                value_index,
                used,
            )
            if hit is None:
                continue
            column_name, value, score = hit
            exact = normalize_text(value) == ngram
            if not exact and score < max(self.value_min_score, 0.93):
                continue
            if not exact and len(ngram) < 4:
                continue
            filters.append(
                QueryFilter(
                    column=column_name,
                    op="eq",
                    value=value,
                    score=score,
                    validated=score >= self.value_min_score,
                    requested_value=ngram,
                )
            )
            used.add(column_name)
            consumed.update(normalize_text(value).split())
            consumed.update(ngram.split())
        return filters

    def _unique_column_value(
        self,
        requested: str,
        dim_columns: Sequence[dict],
        value_index: Dict[str, List[Any]],
        used: set,
    ) -> Optional[Tuple[str, Any, float]]:
        hits = []
        for column in dim_columns:
            name = str(column.get("name") or "")
            if not name or name in used:
                continue
            matched = self._match_value(
                requested,
                value_index.get(name) or [],
            )
            if matched is None:
                continue
            hits.append((matched[1], name, matched[0]))
        if not hits:
            return None
        hits.sort(key=lambda item: item[0], reverse=True)
        if len(hits) > 1 and abs(hits[0][0] - hits[1][0]) < 0.05:
            return None
        return hits[0][1], hits[0][2], hits[0][0]

    def _dedupe(self, filters: List) -> List:
        result = []
        seen = set()
        for item in filters:
            key = (
                item.column,
                item.op,
                normalize_text(str(item.value)),
            )
            if key in seen:
                continue
            seen.add(key)
            result.append(item)
        return result
