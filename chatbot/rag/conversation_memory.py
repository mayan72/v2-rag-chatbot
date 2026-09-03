"""
Lightweight conversation state for clarification and follow-ups.

Stores only the slots needed to resolve the current thread.
Does not dump full chat history into the LLM.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from threading import Lock
from typing import Any, Dict, List, Optional


@dataclass
class ConversationState:
    conversation_id: str
    original_query: str = ""
    last_user_message: str = ""
    clarification: str = ""
    resolved_query: str = ""
    pending_clarification: bool = False
    pending_slot: str = ""
    clarification_question: str = ""
    clarification_options: List[str] = field(default_factory=list)
    clarification_turns: int = 0
    metric: str = ""
    start_period: str = ""
    end_period: str = ""
    last_metric: str = ""
    last_start_period: str = ""
    last_end_period: str = ""
    last_answer: str = ""
    missing_information: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ConversationMemory:
    """In-process memory keyed by conversation_id."""

    def __init__(self, max_conversations: int = 200):
        self._states: Dict[str, ConversationState] = {}
        self._order: List[str] = []
        self._lock = Lock()
        self._max = max_conversations

    def get(self, conversation_id: str) -> ConversationState:
        if not conversation_id:
            return ConversationState(conversation_id="")
        with self._lock:
            state = self._states.get(conversation_id)
            if state is None:
                state = ConversationState(conversation_id=conversation_id)
                self._states[conversation_id] = state
                self._order.append(conversation_id)
                self._evict()
            return state

    def save(self, state: ConversationState) -> None:
        if not state.conversation_id:
            return
        with self._lock:
            self._states[state.conversation_id] = state
            if state.conversation_id in self._order:
                self._order.remove(state.conversation_id)
            self._order.append(state.conversation_id)
            self._evict()

    def clear(self, conversation_id: str) -> None:
        with self._lock:
            self._states.pop(conversation_id, None)
            if conversation_id in self._order:
                self._order.remove(conversation_id)

    def _evict(self) -> None:
        while len(self._order) > self._max:
            oldest = self._order.pop(0)
            self._states.pop(oldest, None)


MEMORY = ConversationMemory()


def get_memory() -> ConversationMemory:
    return MEMORY
