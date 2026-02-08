"""
Corrective-RAG LangGraph agent.

This module builds and compiles the StateGraph and exports it as ``graph``
so that both the FastAPI server and the LangGraph CLI (via langgraph.json)
can reference it with a single import path.

Graph topology
--------------
START
  └─► retrieve
        └─► eval_each_doc
              ├─[CORRECT]──────────────► refine ─► generate ─► END
              └─[INCORRECT/AMBIGUOUS]─► rewrite_query ─► web_search ─► refine ─► generate ─► END
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from my_agent.utils.nodes import (
    eval_each_doc_node,
    generate,
    refine,
    retrieve_node,
    rewrite_query_node,
    route_after_eval,
    web_search_node,
)
from my_agent.utils.state import State

# ── Build graph ────────────────────────────────────────────────────────────
_builder = StateGraph(State)

_builder.add_node("retrieve", retrieve_node)
_builder.add_node("eval_each_doc", eval_each_doc_node)
_builder.add_node("rewrite_query", rewrite_query_node)
_builder.add_node("web_search", web_search_node)
_builder.add_node("refine", refine)
_builder.add_node("generate", generate)

# ── Edges ──────────────────────────────────────────────────────────────────
_builder.add_edge(START, "retrieve")
_builder.add_edge("retrieve", "eval_each_doc")

_builder.add_conditional_edges(
    "eval_each_doc",
    route_after_eval,
    {
        "refine": "refine",
        "rewrite_query": "rewrite_query",
    },
)

# Non-correct path: rewrite → web search → refine
_builder.add_edge("rewrite_query", "web_search")
_builder.add_edge("web_search", "refine")

# Correct path already goes to refine via conditional edge
_builder.add_edge("refine", "generate")
_builder.add_edge("generate", END)

# ── Compile ────────────────────────────────────────────────────────────────
# ``graph`` is the compiled object referenced by langgraph.json and the
# FastAPI server.
graph = _builder.compile()
