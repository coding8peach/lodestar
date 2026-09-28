"""The Lodestar workflow graph.

    START --url--> ingest -> analyze -> review -> END
    START --job_id-----------> analyze            (lodestar review: ingest skipped)
    any node that sets `error` -> END

- ingest:  MCP ingest_url (saves the job, status queued)
- analyze: fit agent over A2A + Python scoring; reuses an earlier analysis if there is one
- review:  pauses with interrupt() so the caller can show the result and ask the
           candidate; resumes with {"decision": "approve" | "reject" | "skip", "note": ...}

Nodes return partial state; any node that fails sets `error`, which routes to END.
The job's status in SQLite is the durable record of where it is, so a run that is
abandoned mid-review simply asks again next time (no LLM call).
Dependencies are injected so tests can replace the MCP call and the fit agent.
"""

import logging
from collections.abc import Awaitable, Callable
from contextlib import closing
from typing import Literal, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from lodestar.db import (
    DecisionError,
    connect,
    get_decision,
    get_status,
    record_decision,
    set_status,
)

# The only fit_agent import: the thin A2A client. No ADK, MCP toolsets or model config.
from lodestar.fit_agent.client import analyze_via_a2a
from lodestar.workflow.analysis import AnalyzeFn, analyze_one
from lodestar.workflow.ingest_client import mcp_ingest

log = logging.getLogger(__name__)

IngestFn = Callable[[str], Awaitable[dict]]
Decision = Literal["approve", "reject", "skip"]


class WorkflowState(TypedDict, total=False):
    url: str
    run_id: int                  # the runs row this invocation's usage is recorded under
    job_id: str
    ingest_status: str           # "saved" | "already_saved"
    fit_result_id: int
    fit_result: dict             # FitResult as JSON-ready data (checkpoints store plain data)
    model: str | None
    reused_analysis: bool        # True when an earlier analysis was loaded instead of running the agent
    decision: Decision
    note: str | None
    already_decided: bool        # True when the decision was made in an earlier run
    error: str


def _error(stage: str, e: Exception) -> dict:
    first_line = (str(e).splitlines() or [""])[0]
    log.error("%s failed: %s: %s", stage, type(e).__name__, first_line[:300])
    return {"error": f"{stage} failed: {first_line[:300] or type(e).__name__}"}


def build_graph(
    ingest_fn: IngestFn = mcp_ingest,
    analyze_fn: AnalyzeFn = analyze_via_a2a,
    checkpointer=None,
):
    async def ingest(state: WorkflowState) -> dict:
        try:
            out = await ingest_fn(state["url"])
        except Exception as e:
            return _error("ingest", e)
        log.info("ingest: %s %s (%s)", out["status"], out["job_id"], out.get("title"))
        return {"job_id": out["job_id"], "ingest_status": out["status"]}

    async def analyze(state: WorkflowState) -> dict:
        out = await analyze_one(state["job_id"], state.get("run_id"), analyze_fn)
        out.pop("error_kind", None)
        return out

    async def review(state: WorkflowState) -> dict:
        # LangGraph re-runs this node from the top when it resumes, so everything
        # before interrupt() must be safe to run twice.
        job_id, fit_id = state["job_id"], state["fit_result_id"]
        with closing(connect()) as conn:
            status = get_status(conn, job_id)
            if status in ("approved", "rejected"):
                earlier = get_decision(conn, fit_id) or {}
                log.info("review: already %s", status)
                return {"decision": earlier.get("decision"), "note": earlier.get("note"), "already_decided": True}
            if status == "analyzed":
                set_status(conn, job_id, "awaiting_review")

        answer = interrupt({"job_id": job_id, "fit_result_id": fit_id})

        decision, note = answer.get("decision"), (answer.get("note") or None)
        try:
            with closing(connect()) as conn:
                record_decision(conn, job_id, decision, note)
        except DecisionError as e:
            return {"error": f"review failed: {e}"}
        if decision == "skip":
            log.info("review: skipped; job stays awaiting_review")
        return {"decision": decision, "note": note, "already_decided": False}

    def ok_or_end(next_node: str) -> Callable[[WorkflowState], str]:
        return lambda state: END if state.get("error") else next_node

    graph = StateGraph(WorkflowState)
    graph.add_node("ingest", ingest)
    graph.add_node("analyze", analyze)
    graph.add_node("review", review)
    # A url starts at ingest; a job_id (e.g. `lodestar review`) goes straight to analyze,
    # which reuses the stored analysis, then to the same review node.
    graph.add_conditional_edges(START, lambda state: "analyze" if state.get("job_id") else "ingest",
                                ["ingest", "analyze"])
    graph.add_conditional_edges("ingest", ok_or_end("analyze"), ["analyze", END])
    graph.add_conditional_edges("analyze", ok_or_end("review"), ["review", END])
    graph.add_edge("review", END)
    # interrupt() needs a checkpointer to hold the paused state. v0 pauses and resumes
    # within one process, so memory is enough; SQLite status is the durable record.
    return graph.compile(checkpointer=checkpointer or InMemorySaver())
