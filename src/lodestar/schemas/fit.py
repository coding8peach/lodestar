"""Fit schemas. The agent returns FitAnalysis (no score); Python computes FitResult."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel


class RequirementMatch(BaseModel):
    requirement: str
    priority: Literal["must_have", "nice_to_have"]
    evidence: list[str]  # from the profile; empty for a gap
    match_level: Literal["direct", "related", "gap"]
    explanation: str
    # Location, residency, work authorization, time zone, required degree or clearance.
    # The agent decides what counts (semantic); scoring.py applies the consequence (arithmetic).
    hard_constraint: bool = False


class FitAnalysis(BaseModel):  # what the ADK agent returns. NO score.
    job_id: str
    requirements: list[RequirementMatch]
    summary: str


Recommendation = Literal["top_pick", "possible", "blocked"]


class FitResult(BaseModel):  # what gets stored. Score computed in Python.
    job_id: str
    analysis: FitAnalysis
    overall_score: float
    recommendation: Recommendation  # top_pick: apply first; possible: worth a look; blocked: can't apply


class LlmCall(BaseModel):
    """One model call made during an analysis, successful or not. Usage metadata only."""

    model: str
    attempt: int                 # position in the fallback order: 1 = first model tried
    status: Literal["ok", "rate_limited", "error"]
    input_tokens: int = 0
    output_tokens: int = 0       # billed output tokens, thinking included
    thinking_tokens: int = 0     # of which thinking (informational)
    latency_ms: int | None = None
    error: str | None = None
    started_at: datetime


class FitAgentReply(BaseModel):  # what the fit agent service returns over A2A. Still NO score.
    analysis: FitAnalysis
    model: str                   # the one model that produced the analysis
    skipped: list[str] = []      # models that fell through (rate limit, size, outage), with reasons
    tokens: int | None = None    # total tokens reported for the successful attempt
    calls: list[LlmCall] = []    # every model call, including failed attempts that fell through
