"""
Node functions and routing logic for the Corrective-RAG LangGraph agent.

Graph flow
----------
START
  └─► retrieve
        └─► eval_each_doc
              ├─[CORRECT]──────────────► refine ─► generate ─► END
              └─[INCORRECT/AMBIGUOUS]─► rewrite_query ─► web_search ─► refine ─► generate ─► END

Routing behaviour
-----------------
CORRECT   : at least one doc scores > UPPER_TH (0.7)  → use only local docs
INCORRECT : all docs score < LOWER_TH (0.3)           → use only web docs
AMBIGUOUS : mixed scores                               → merge local + web docs
"""

from __future__ import annotations

import re
from typing import List

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_deepseek import ChatDeepSeek
from pydantic import BaseModel

from my_agent.utils.state import State
from my_agent.utils.tools import retriever, tavily

load_dotenv()

# ── Thresholds ─────────────────────────────────────────────────────────────
UPPER_TH = 0.7
LOWER_TH = 0.3

# ── LLM ────────────────────────────────────────────────────────────────────
llm = ChatDeepSeek(
    model="deepseek-v4-flash", extra_body={"thinking": {"type": "disabled"}}
)


# ══════════════════════════════════════════════════════════════════════════
# Node: retrieve
# ══════════════════════════════════════════════════════════════════════════


def retrieve_node(state: State) -> dict:
    """Retrieve the top-k documents from the FAISS index."""
    return {"docs": retriever.invoke(state["question"])}


# ══════════════════════════════════════════════════════════════════════════
# Node: eval_each_doc
# ══════════════════════════════════════════════════════════════════════════


class DocEvalScore(BaseModel):
    score: float
    reason: str


doc_eval_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a strict retrieval evaluator for RAG.\n"
            "You will be given ONE retrieved chunk and a question.\n"
            "Return a relevance score in [0.0, 1.0].\n"
            "- 1.0: chunk alone is sufficient to answer fully/mostly\n"
            "- 0.0: chunk is irrelevant\n"
            "Be conservative with high scores.\n"
            "Also return a short reason.\n"
            "Output JSON only.",
        ),
        ("human", "Question: {question}\n\nChunk:\n{chunk}"),
    ]
)

doc_eval_chain = doc_eval_prompt | llm.with_structured_output(DocEvalScore)


def eval_each_doc_node(state: State) -> dict:
    """Score every retrieved chunk and decide the retrieval verdict."""
    q = state["question"]
    scores: List[float] = []
    good: List[Document] = []

    for doc in state["docs"]:
        out: DocEvalScore = doc_eval_chain.invoke(
            {"question": q, "chunk": doc.page_content}
        )
        scores.append(out.score)
        if out.score > LOWER_TH:
            good.append(doc)

    # ── verdict logic ──────────────────────────────────────────────────
    if any(s > UPPER_TH for s in scores):
        return {
            "good_docs": good,
            "verdict": "CORRECT",
            "reason": f"At least one retrieved chunk scored > {UPPER_TH}.",
        }

    if scores and all(s < LOWER_TH for s in scores):
        return {
            "good_docs": [],
            "verdict": "INCORRECT",
            "reason": f"All retrieved chunks scored < {LOWER_TH}.",
        }

    return {
        "good_docs": good,
        "verdict": "AMBIGUOUS",
        "reason": f"No chunk scored > {UPPER_TH}, but not all were < {LOWER_TH}.",
    }


# ══════════════════════════════════════════════════════════════════════════
# Node: rewrite_query
# ══════════════════════════════════════════════════════════════════════════


class WebQuery(BaseModel):
    query: str


rewrite_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Rewrite the user question into a web search query composed of keywords.\n"
            "Rules:\n"
            "- Keep it short (6–14 words).\n"
            "- If the question implies recency (e.g., recent/latest/last week/last month),"
            " add a constraint like (last 30 days).\n"
            "- Do NOT answer the question.\n"
            "- Return JSON with a single key: query",
        ),
        ("human", "Question: {question}"),
    ]
)

rewrite_chain = rewrite_prompt | llm.with_structured_output(WebQuery)


def rewrite_query_node(state: State) -> dict:
    """Rewrite the question into a terse web-search query."""
    out: WebQuery = rewrite_chain.invoke({"question": state["question"]})
    return {"web_query": out.query}


# ══════════════════════════════════════════════════════════════════════════
# Node: web_search
# ══════════════════════════════════════════════════════════════════════════


def web_search_node(state: State) -> dict:
    """Run a Tavily web search and convert results to Documents."""
    q = state.get("web_query") or state["question"]
    raw_results = tavily.invoke({"query": q})

    # TavilySearch returns a dict with a 'results' list, or sometimes a list of dicts directly
    if isinstance(raw_results, dict):
        results_list = raw_results.get("results", [])
    elif isinstance(raw_results, list):
        results_list = raw_results
    else:
        results_list = []

    web_docs: List[Document] = []
    for r in results_list:
        if isinstance(r, dict):
            title = r.get("title", "")
            url = r.get("url", "")
            content = r.get("content", "") or r.get("snippet", "")
            text = f"TITLE: {title}\nURL: {url}\nCONTENT:\n{content}"
            web_docs.append(
                Document(page_content=text, metadata={"url": url, "title": title})
            )

    return {"web_docs": web_docs}


# ══════════════════════════════════════════════════════════════════════════
# Node: refine
# ══════════════════════════════════════════════════════════════════════════


class KeepOrDrop(BaseModel):
    keep: bool


filter_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a strict relevance filter.\n"
            "Return keep=true only if the sentence directly helps answer the question.\n"
            "Use ONLY the sentence. Output JSON only.",
        ),
        ("human", "Question: {question}\n\nSentence:\n{sentence}"),
    ]
)

filter_chain = filter_prompt | llm.with_structured_output(KeepOrDrop)


def decompose_to_sentences(text: str) -> List[str]:
    """Split a block of text into individual sentences (≥20 chars)."""
    text = re.sub(r"\s+", " ", text).strip()
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if len(s.strip()) > 20]


def refine(state: State) -> dict:
    """
    Filter retrieved context down to sentences that directly answer the question.

    CORRECT   → use only local good_docs
    INCORRECT → use only web_docs
    AMBIGUOUS → merge good_docs + web_docs
    """
    q = state["question"]

    if state.get("verdict") == "CORRECT":
        docs_to_use = state["good_docs"]
    elif state.get("verdict") == "INCORRECT":
        docs_to_use = state.get("web_docs", [])
    else:  # AMBIGUOUS
        docs_to_use = state.get("good_docs", []) + state.get("web_docs", [])

    context = "\n\n".join(d.page_content for d in docs_to_use).strip()
    strips = decompose_to_sentences(context)

    kept: List[str] = []
    for sentence in strips:
        out: KeepOrDrop = filter_chain.invoke({"question": q, "sentence": sentence})
        if out.keep:
            kept.append(sentence)

    return {
        "strips": strips,
        "kept_strips": kept,
        "refined_context": "\n".join(kept).strip(),
    }


# ══════════════════════════════════════════════════════════════════════════
# Node: generate
# ══════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = (
    "You are a helpful ML tutor. Answer ONLY using the provided context.\n"
    "If the context is empty or insufficient, say: 'I don't know.'"
)


def generate(state: State) -> dict:
    """
    Generate the final answer, optionally enriched by prior chat history.

    Chat history is injected between the system message and the final
    human message so the model is aware of the conversation context.
    """
    messages = [SystemMessage(content=SYSTEM_PROMPT)]

    # Inject prior conversation turns (if any)
    for msg in state.get("chat_history", []):
        messages.append(msg)

    # Final user turn with context
    messages.append(
        HumanMessage(
            content=(
                f"Question: {state['question']}\n\nContext:\n{state['refined_context']}"
            )
        )
    )

    response = llm.invoke(messages)
    return {"answer": response.content}


# ══════════════════════════════════════════════════════════════════════════
# Routing
# ══════════════════════════════════════════════════════════════════════════


def route_after_eval(state: State) -> str:
    """
    Direct the graph after document evaluation:
    - CORRECT  → straight to refinement (no web search needed)
    - otherwise → rewrite query, then web search, then refine
    """
    if state["verdict"] == "CORRECT":
        return "refine"
    return "rewrite_query"
