"""
State definition for the Corrective-RAG LangGraph agent.

The TypedDict carries every intermediate value the graph needs across
nodes.  ``chat_history`` holds the conversation turns that preceded the
current query so the generate node can use them as context.
"""

from __future__ import annotations

from typing import List

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict


class State(TypedDict):
    # ── input ─────────────────────────────────────────────────────────
    question: str
    # conversation turns that arrived with the request (human / ai msgs)
    chat_history: List[BaseMessage]

    # ── retrieval ─────────────────────────────────────────────────────
    docs: List[Document]
    good_docs: List[Document]

    # ── doc-evaluation verdict ────────────────────────────────────────
    verdict: str   # "CORRECT" | "INCORRECT" | "AMBIGUOUS"
    reason: str

    # ── knowledge refinement ──────────────────────────────────────────
    strips: List[str]
    kept_strips: List[str]
    refined_context: str

    # ── web search ────────────────────────────────────────────────────
    web_query: str
    web_docs: List[Document]

    # ── output ────────────────────────────────────────────────────────
    answer: str
