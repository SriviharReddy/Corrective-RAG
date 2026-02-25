"""
FastAPI microservice for the Corrective-RAG LangGraph agent.

Endpoints
---------
POST /chat
    Accept a query + optional chat history, run the Corrective-RAG graph,
    and return the structured answer.

GET /health
    Simple liveness check.

Usage
-----
    uv run uvicorn api.server:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

from typing import List, Literal, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, Field

load_dotenv()

# Import the compiled graph (also triggers FAISS index build at startup)
from my_agent.agent import graph  # noqa: E402  (after load_dotenv)

# ── Pydantic schemas ────────────────────────────────────────────────────────

class ChatMessage(BaseModel):
    """A single turn in the conversation history."""

    role: Literal["human", "ai"] = Field(
        ...,
        description="Speaker role: 'human' for user messages, 'ai' for assistant messages.",
    )
    content: str = Field(..., description="Text content of the message.")


class ChatRequest(BaseModel):
    """Request body for POST /chat."""

    query: str = Field(..., description="The current user question.")
    chat_history: List[ChatMessage] = Field(
        default_factory=list,
        description=(
            "Ordered list of prior conversation turns. "
            "Pass the last N exchanges to give the model context."
        ),
    )


class ChatResponse(BaseModel):
    """Response body from POST /chat."""

    answer: str = Field(..., description="The generated answer.")
    verdict: str = Field(
        ...,
        description="Retrieval verdict: CORRECT | INCORRECT | AMBIGUOUS.",
    )
    reason: str = Field(..., description="Explanation of the verdict.")
    web_query: Optional[str] = Field(
        None,
        description="The rewritten web-search query (populated when verdict ≠ CORRECT).",
    )


# ── FastAPI app ─────────────────────────────────────────────────────────────

app = FastAPI(
    title="Corrective-RAG Agent",
    description=(
        "A Corrective-RAG microservice backed by LangGraph. "
        "Retrieves from a local PDF index, evaluates chunk relevance, "
        "optionally supplements with live web search, then generates an answer."
    ),
    version="1.0.0",
)


@app.get("/health", summary="Liveness check")
async def health() -> dict:
    """Return 200 OK when the service is running."""
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse, summary="Run the Corrective-RAG agent")
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Execute the Corrective-RAG graph for a single query.

    The ``chat_history`` is converted to LangChain ``HumanMessage`` /
    ``AIMessage`` objects and injected into the state so the generate node
    can use prior turns as context.
    """
    # Convert chat history to LangChain message objects
    lc_history = []
    for msg in request.chat_history:
        if msg.role == "human":
            lc_history.append(HumanMessage(content=msg.content))
        else:
            lc_history.append(AIMessage(content=msg.content))

    # Build the initial state (all list/string fields need defaults)
    initial_state = {
        "question": request.query,
        "chat_history": lc_history,
        "docs": [],
        "good_docs": [],
        "verdict": "",
        "reason": "",
        "strips": [],
        "kept_strips": [],
        "refined_context": "",
        "web_query": "",
        "web_docs": [],
        "answer": "",
    }

    try:
        result = await graph.ainvoke(initial_state)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return ChatResponse(
        answer=result.get("answer", ""),
        verdict=result.get("verdict", ""),
        reason=result.get("reason", ""),
        web_query=result.get("web_query") or None,
    )
