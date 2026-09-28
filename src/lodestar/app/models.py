"""View models returned by the service layer. They would become the API schemas of a
FastAPI layer, unchanged."""

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel

from lodestar.db.status import JobStatus
from lodestar.schemas.fit import FitResult, Recommendation
from lodestar.schemas.job import JobPosting


class ReviewItem(BaseModel):
    job_id: str
    company: str | None
    title: str
    location: str | None
    url: str
    recommendation: Recommendation
    score: float
    blocked_by: list[str] = []          # unmet must-have hard constraints
    status: JobStatus
    analyzed_at: datetime


class Decision(BaseModel):
    decision: Literal["approve", "reject"]
    note: str | None
    decided_at: datetime


class SimilarJob(BaseModel):
    job_id: str
    title: str
    status: JobStatus


class JobDetail(BaseModel):
    job: JobPosting
    status: JobStatus
    fit: FitResult | None = None
    fit_result_id: int | None = None
    model: str | None = None
    analyzed_at: datetime | None = None
    blocked_by: list[str] = []
    decision: Decision | None = None
    dismissed_reason: str | None = None
    similar: list[SimilarJob] = []      # near-duplicates: same company, same title without seniority


class QueueItem(BaseModel):
    job_id: str
    company: str | None
    title: str
    location: str | None
    url: str
    score: int
    matched: list[str]
    similar_count: int
    date_posted: date | None


class Funnel(BaseModel):
    by_status: dict[str, int]           # jobs per lifecycle status
    discovered: int                     # postings seen by `discover`
    discovery_skipped: int              # dropped by the pre-filter (incl. invalid postings)

    def count(self, *statuses: str) -> int:
        return sum(self.by_status.get(s, 0) for s in statuses)


class DecisionItem(BaseModel):
    job_id: str
    company: str | None
    title: str
    url: str
    decision: Literal["approve", "reject"]
    note: str | None
    decided_at: datetime
    recommendation: Recommendation | None   # None when dismissed without analysis
    score: float | None
    dismissed: bool = False


class UsageRow(BaseModel):
    key: str                            # run ("#4 batch 2026-09-28T07:24"), day, or model
    calls: int
    failed_calls: int
    input_tokens: int
    output_tokens: int
    list_cost_usd: float | None
    cost_usd: float | None
    unpriced_calls: int


class BulkResult(BaseModel):
    done: int
    not_done: dict[str, str] = {}       # job_id -> why it was left alone


class BoardSummary(BaseModel):
    company: str
    listed: int
    seen: int
    queued: int
    skipped: dict[str, int] = {}        # reason -> count
    error: str | None = None


class DiscoverySummary(BaseModel):
    boards: list[BoardSummary]
    queued_jobs: list[tuple[str | None, str, str | None]]   # (company, title, location)
    dry_run: bool

    @property
    def listed(self) -> int:
        return sum(b.listed for b in self.boards)

    @property
    def seen(self) -> int:
        return sum(b.seen for b in self.boards)

    @property
    def skipped(self) -> dict[str, int]:
        total: dict[str, int] = {}
        for b in self.boards:
            for reason, n in b.skipped.items():
                total[reason] = total.get(reason, 0) + n
        return dict(sorted(total.items(), key=lambda kv: -kv[1]))


class AnalyzedJob(BaseModel):
    job_id: str
    company: str | None
    title: str
    recommendation: Recommendation | None = None
    score: float | None = None
    error: str | None = None


class BatchResult(BaseModel):
    run_id: int | None
    results: list[AnalyzedJob] = []
    stopped: str | None = None          # why the batch ended early, if it did
    paid_limit_usd: float = 0.0

    @property
    def ok(self) -> int:
        return sum(1 for r in self.results if r.error is None)

    @property
    def failed(self) -> int:
        return sum(1 for r in self.results if r.error is not None)


class Spending(BaseModel):
    paid_limit_usd: float               # LODESTAR_PAID_USD_PER_DAY
    paid_today_usd: float               # actual spend today (free tiers count as $0)
