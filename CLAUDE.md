# Lodestar

An AI job-search platform. It ingests job postings, judges how well each one fits
the candidate, lets the candidate review the results, and (later) tailors the
resume for approved jobs. This is a restart (v2) of an earlier version, rebuilt
around LangGraph, Google ADK, A2A, and MCP, where each technology has a real job.

**Status:** v0 complete (one round trip). v0.x (fit quality) mostly done. v1 first slice
complete (watchlist discovery → pre-filter → ranking → budgeted analysis → review, usage
and cost tracking). Local Streamlit UI with the whole loop. v2 first slice complete (resume agent). Demo mode
built. Goal: deploy the read-only demo as a portfolio project (GitHub: coding8peach/lodestar).

The daily loop, in the UI:

```bash
uv run lodestar-agent        # terminal 1: fit agent service (A2A, port 8001)
uv run lodestar ui           # terminal 2: browser UI at localhost:8501
```

Queue → Find new postings (free) → Analyze next N → Review → decide → Decisions
(Approved = apply list) → select a job → Tailor resume → download .docx / .md. The same steps from the terminal: `lodestar discover`,
`lodestar analyze`, `lodestar review`, `lodestar usage`.

## Architecture

| Technology | Role |
|---|---|
| LangGraph | Main workflow: ingest → analyze → review, with human review via `interrupt()` |
| Google ADK | Runtime for the fit agent (`LlmAgent` + `McpToolset`), served with `to_a2a` |
| A2A | The workflow asks the fit agent service for an analysis (job_id in, JSON out) |
| MCP | Lodestar MCP server: `ingest_url`, `get_job`, `get_profile` over stdio |
| LLM | Semantic judgment only: requirement extraction and per-requirement fit |
| Python | Deterministic logic: classification rules, validation, scoring math, fallback |
| SQLite | Jobs, fit results, requirement matches, decisions (WAL + busy_timeout) |
| Pydantic | All schemas |

MCP connects agent to tool. A2A connects agent to agent.

**LLM → semantic judgment. Python → arithmetic.**

```text
LangGraph workflow (uv run lodestar URL)
  ingest ──MCP──► Lodestar MCP server ──► SQLite
  analyze ──A2A──► fit agent service (uv run lodestar-agent, port 8001)
                     └─ FitService → analyze_job (per-run fallback) → LlmAgent
                          └─MCP──► get_job, get_profile
          ◄── FitAgentReply → Python scoring → SQLite
  review ── interrupt: terminal shows the fit table, approve / reject / skip → SQLite
```

## Job lifecycle

discovered → deduplicated → queued → analyzed → awaiting_review
→ approved / rejected → resume_tailored → ready_to_apply → applied

Used today: queued → analyzed → awaiting_review → approved / rejected, and
queued → rejected (dismissed without analysis; reason in `jobs.dismissed_reason`).
Allowed moves are enforced in `db/status.py` with a conditional UPDATE. No re-analysis
transition yet (analyzed → queued).

## Key design decisions

- **Fit is the core feature.** The fit agent assesses each requirement
  separately: priority (must_have / nice_to_have), evidence (profile entry ids),
  and a match level (direct / related / gap). "Related" matters: adjacent
  experience counts. Postings are wish lists; a missing nice-to-have must not
  sink a job, and a single must-have gap should not automatically sink one either.
- **Hard constraints gate; skills score.** The agent marks hard constraints
  (location, residency, work authorization, time zone, required degree/clearance)
  with `hard_constraint: true`. They are left out of the score. An unmet must-have
  hard constraint makes the job `blocked`; nothing else does.
- **Recommendations:** `top_pick` (score ≥ `TOP_PICK_MIN`), `possible` (everything
  else not blocked, whatever the score: not ruled out), `blocked` (can't apply).
  The score still orders jobs. This matches how the candidate decides: low skill
  fit means "worth a look", only a blocker means "no".
- **The agent never produces a score.** Its output schema contains only
  requirement matches and a summary. Python computes the score and
  recommendation afterward. (If the output schema has a score field, the LLM
  will fill it in and we get two scores that disagree.)
- **Evidence is profile entry ids** (`exp-...`, `proj-...`, `edu-...`,
  `course-...`), so Python can check every citation exists.
- **The candidate's approve/reject decision is stored next to the agent's
  assessment**, so agreement can be measured over time.
- **Profile vs resume:** the profile is everything about the candidate (for the
  agent); a resume is a curated selection (for employers). Skills carry depth:
  production / project / course. One entry per skill: `depth` = deepest level
  ever reached, `last_used` = latest year at any level.
- **Profile files:** `data/profile.yaml` (gitignored), hand-written in v0 and
  validated on load (evidence ids must exist, unknown keys fail). Later, a parser
  writes `profile.draft.yaml`; re-parsing only overwrites the draft.
- **Job ingestion uses adapters.** Every source produces the same normalized
  `JobPosting`. Nothing downstream cares about the source.
- **Known ATS URLs are fetched through public job-board APIs**, not by parsing the
  page (ATS pages don't reliably include JSON-LD): Greenhouse
  `boards-api.greenhouse.io`, Lever `api.lever.co`, Ashby `api.ashbyhq.com`.
  Ashby and Lever APIs omit the company name, so it is read from the hosted
  page's `<title>`; that lookup never blocks ingestion.
- **Page classification is deterministic first:**
  1. URL rules for single-posting URLs (board index and /apply pages don't match;
     `/application` and `/apply` suffixes are trimmed)
  2. schema.org `JobPosting` JSON-LD in the page
  3. LLM fallback only when neither decides (**deferred**: not built yet)
  Each posting records which layer decided (`classified_by`).
- **Job ids** are a hash of the normalized URL (only known tracking params are
  stripped; e.g. `gh_jid` is kept). Same URL → same id → re-ingesting is a no-op.
- **Don't use an agent where a pipeline step will do.**
- **Don't scrape Indeed or LinkedIn.** Discovery sources: Serper Google search
  restricted to ATS sites, Greenhouse/Lever/Ashby public board APIs, pasted URLs.

## The fit agent service

Layers, outside in (`src/lodestar/fit_agent/`):

- `server.py`: `to_a2a(FitService)` under uvicorn. Port `LODESTAR_AGENT_PORT`
  (default 8001), passed to both to_a2a and uvicorn so the agent card is right.
- `service.py` — `FitService`, a custom ADK `BaseAgent` with no LLM of its own:
  finds the job_id in the message, checks it via MCP `get_job`, runs
  `analyze_job`, replies with one `FitAgentReply` JSON message. Errors → failed
  A2A task with the message.
- `runner.py` — `analyze_job`: **per-run fallback.** Tries models in order
  (`LODESTAR_FIT_MODELS`); each attempt is a fresh agent, session and MCP
  connection on one model. Rate limit / size limit / outage at any turn → start
  over on the next model. Other errors are raised. Bad JSON → one retry on the
  same model. A temporary overload (503 "high demand", 529) first retries the same model
  after 5 s and 15 s (`OVERLOAD_RETRY_DELAYS`); rate limits and quotas (429) fall back
  immediately. Every attempt is recorded in usage.
- `agent.py` — the `LlmAgent`: instruction, `McpToolset` limited to `get_job` and
  `get_profile`, and `compact_tool_result` (MCP results carry the data twice;
  keep the structured copy, drop empty and bookkeeping fields: ~65% smaller).
- `models.py` — Gemini natively (keeps its thought signatures), everything else
  via LiteLLM with earlier reasoning stripped from the history (Groq rejects it).
- `usage.py` — before/after/on-error model callbacks record every call (tokens,
  latency, status incl. rate limits). Returned in `FitAgentReply.calls`; the service
  never writes the database, callers store usage.
- Paid budget: callers send `max_paid_usd` as A2A request metadata; at 0 the service
  drops paid models (only `free_tier` models in `config/prices.yaml`; unknown price =
  paid) so a free-model rate limit ends the run instead of spending money.
- `parse.py` — reply → `FitAnalysis`; flags evidence ids not in the profile.
- `client.py` — `analyze_via_a2a(job_id)`: all a caller needs.

**Why per-run, not per-call, fallback:** each provider leaves its own artifacts in
a conversation (reasoning fields, thought signatures). A history started by one
provider can't safely be continued by another. Fallback happens where the state
is neutral: the start of a run, where only a job_id crosses.

## Schemas (`src/lodestar/schemas/`)

`Profile` (targets, experience, projects, education, skills, languages),
`JobPosting` (id, url, source, classified_by, title, company, location,
work_mode, seniority, salary_text, date_posted, valid_through, description,
fetched_at; no requirements, no status), and:

```python
class RequirementMatch(BaseModel):
    requirement: str
    priority: Literal["must_have", "nice_to_have"]
    evidence: list[str]          # profile entry ids; empty for a gap and for hard constraints
    match_level: Literal["direct", "related", "gap"]
    explanation: str
    hard_constraint: bool = False  # location, work authorization, ... (agent decides; Python applies)

class FitAnalysis(BaseModel):    # the agent's judgment. NO score.
    job_id: str
    requirements: list[RequirementMatch]
    summary: str

class FitAgentReply(BaseModel):  # what the service returns over A2A. Still NO score.
    analysis: FitAnalysis
    model: str                   # the one model that produced it
    skipped: list[str] = []      # models that fell through, with reasons
    tokens: int | None = None
    calls: list[LlmCall] = []    # every model call incl. failed attempts (usage metadata)

class FitResult(BaseModel):      # what gets stored. Score computed in Python.
    job_id: str
    analysis: FitAnalysis
    overall_score: float
    recommendation: Literal["top_pick", "possible", "blocked"]
```

Scoring (`scoring.py`): over non-hard-constraint requirements only, direct = 1.0,
related = 0.6, gap = 0.0; must_have weight 2, nice_to_have weight 1; score = weighted
average. blocked if an unmet must-have hard constraint; else top_pick if score ≥
`TOP_PICK_MIN` (0.70, provisional); else possible.

## Database (`data/lodestar.sqlite`, schema v6)

```text
jobs                 id, url, source, classified_by, title, company, location, work_mode,
                     seniority, salary_text, date_posted, valid_through, description,
                     fetched_at, status, created_at, updated_at, dismissed_reason
fit_results          id, job_id, overall_score, recommendation, summary, model, created_at, run_id
requirement_matches  id, fit_result_id, position, requirement, priority, match_level,
                     evidence (JSON), explanation, hard_constraint
decisions            id, fit_result_id, decision (approve/reject), note, created_at
runs                 id, kind (workflow/eval/batch), label, started_at, finished_at,
                     analyses_ok, analyses_failed
llm_calls            id, run_id, job_id, fit_result_id, model, attempt, status, input/output/
                     thinking tokens, latency_ms, error, list_cost_usd, cost_usd, price_source
discoveries          id, job_id (unique), url, company, ats, board, title, location, work_mode,
                     verdict (queued/skipped/invalid/known), reason, first_seen_at, last_seen_at
```

CHECK constraints mirror the Pydantic literals; foreign keys on. Migrations in
`db/connection.py` run automatically on open (v2 hard_constraint; v3 recommendations
rebuilt and recomputed; v4 usage tables; v5 discoveries; v6 dismissed_reason). Back up
`data/lodestar.sqlite` before upgrading. `requirement_matches` as its own table makes
the future gap report a simple `GROUP BY`.

## Ingestion (`src/lodestar/ingest/`)

Only `ingest_url()` is exposed through MCP:

```text
ingest_url(url)            already saved? → return it, no refetch
  ├── build_posting()      known ATS URL → its public API; else fetch → classify → JSON-LD
  ├── validate_job()       trim whitespace; reject descriptions < 200 chars; warn if expired
  └── save_job()           status queued
```

## Discovery (v1): `lodestar discover`, no LLM

`config/watchlist.yaml` lists companies with their ATS and board name (plus a blocklist
for aggregators such as Jobgether). Each board is listed with one request (Greenhouse
`/boards/{b}/jobs?content=true`, Lever `/v0/postings/{b}?mode=json`, Ashby job-board API),
which returns every open job with its description. Jobs are mapped with the same code as
`ingest_url` and stored under their canonical hosted URL, so hand-pasted jobs dedupe.
Company name comes from the watchlist. Retries with backoff on 429/5xx/timeouts; a
failing board is reported and skipped. `discoveries` is the seen-registry: known postings
aren't re-filtered unless `--recheck`. `--dry-run` saves nothing.

Pre-filter (`ingest/prefilter.py`, rules in `config/filters.yaml`, whole-word matching):
engineering titles only; excludes by role type (iOS, manager, intern, customer-facing,
hardware, security/network/IT, frontend, ML/research, analytics, test), below senior
(junior, new grad, "I/II", early career) and above staff (principal, director, head of);
location = Bay Area cities, US remote (remote tied only to non-US places is dropped), or
no location stated (the agent's hard-constraint check decides). Every skip has a reason.
First real run: 796 listed → 80 queued.

## Ranking and budgeted analysis (v1): `lodestar analyze`

`ranking.py`, deterministic, order only (never a verdict): near-duplicate groups (same
company + title without Senior/Staff/Sr.; once one is analyzed the rest are skipped),
relevance score from profile skills (weighted by depth; title ×3; description capped)
minus frontend-heavy terms (`ranking.penalize` in filters.yaml; a penalized term that is
also a skill counts only as a penalty), newest first on ties, `max_per_company_per_run`.
`--budget N` (default 3) analyzes the top N through `workflow/analysis.py::analyze_one`
(the same path the workflow uses), recorded as a `batch` run. Stops early if the agent is
unreachable or every model failed. Paid spend is capped by `LODESTAR_PAID_USD_PER_DAY`
(default 0 = free models only), checked against today's recorded `cost_usd`.

## Usage and cost (v1): `lodestar usage [--by run|day|model]`

Every model call is stored in `llm_calls` with a cost estimate made at save time:
LiteLLM's price table, overridden by `config/prices.yaml` (`free_tier: true` → you pay
$0, list-price estimate still shown). Unknown prices are NULL and reported, never $0.
Measured: one analysis ≈ 2 calls, ~9.5–9.8K input and ~0.7–0.9K output tokens, ~7 s;
≈ $0.0035 list on gemini-3.1-flash-lite (free tier: $0); ≈ $0.003 est. on gpt-5.4-nano.

## Service layer and UI

`src/lodestar/app/`: `models.py` (Pydantic view models: ReviewItem, JobDetail, QueueItem,
Funnel, DecisionItem, UsageRow, BulkResult, DiscoverySummary, BatchResult, Spending) and
`service.py` (plain functions: review_queue, job_detail, record_decision, reject_blocked,
dismiss, ranked_queue, funnel, decisions, usage, spending, discover, analyze_next). This
is the only thing a UI calls; the CLI's discover/analyze/review use it too. Designed so a
FastAPI layer would be a thin wrapper (one endpoint per function, models as schemas).

`record_decision` (db/repo.py) is the single place decisions are saved: workflow review
node, `lodestar review`, and the UI all use it.

`src/lodestar/ui/` (Streamlit, `uv run lodestar ui`, localhost only): Review (filters,
table, detail with colored requirements, Approve/Reject/Skip, blocked section), Queue
(funnel, Find new postings, Analyze next N with agent status and paid spend, ranked queue,
Dismiss), Decisions (Approved = apply list, Rejected incl. dismissed), Usage. Pages call
only `app/service.py`. Retries inside the fit agent aren't shown live (A2A returns once);
streaming progress over A2A would be a separate change. Tests use Streamlit's AppTest.

## Resume tailoring (v2)

A second ADK agent (`resume_agent/`), served by the same `lodestar-agent` process on port 8002
(fit agent on 8001). It reads `get_job`, `get_profile` and `get_fit_analysis` (MCP, read-only)
and returns a `TailoredResume` (schemas/resume.py): headline, summary, skills, experience and
project entries by id with reworded highlights, and notes for the candidate. Every line cites
profile entry ids. It selects, orders and rewords; it never invents.

Python (resume/validate.py) rejects: unknown entries or sources, skills not in the profile,
numbers not in the cited sources ("N+ years" allowed within the career span), claim words
("proven", "high-scale", "expert", ...) not in the sources or profile summary, and
placeholders. One retry lists the problems. Roles since 2006 left out are added back by
Python. Facts (name, contact, titles, companies, dates, education) are filled in from the
profile by id (resume/assemble.py); placeholders in the profile block saving. Rendered to
Markdown and Word (resume/render.py, US Letter, single column). Stored in `resumes`
(schema v7; versions kept); the first resume moves the job approved → resume_tailored.
Both agents share `run_with_fallback` (fit_agent/runner.py). Prompt versions: fit v2c,
resume r2. Limitation: the checks can't catch subtle overstatement in a reworded true line;
the candidate reads every resume.

## Demo mode

`LODESTAR_DEMO=1`: banner on every page; every action that calls an LLM, fetches boards or
writes data is disabled in the UI and refused by the service layer (`DemoReadOnly`). Demo
data: `demo/profile.yaml` (fictional Morgan Lee, mirroring a backend-to-agents career),
`demo/postings.yaml` (16 fictional postings: 12 pass the pre-filter, incl. a UK-only one that
comes out blocked, a frontend-heavy one, and a Senior/Staff near-duplicate pair).
`uv run python demo/build_demo.py` runs the real pre-filter, fit agent and resume agent
in-process and writes `demo/demo.sqlite` (committed; single file).

## Workflow (`src/lodestar/workflow/`)

`START → ingest → analyze → review → END` for a URL; started with a `job_id` (as
`lodestar review` does) it skips ingest. Any node that fails sets `error` and routes to END. `analyze` reuses an existing analysis (no LLM call) unless the job
is `queued`. `review` pauses with `interrupt()`; the CLI shows the table and
resumes with approve / reject / skip. The checkpointer is in-memory: SQLite status
is the durable record, so an abandoned review simply asks again next time.

## Running

```bash
uv run lodestar-agent                 # terminal 1: fit agent service (A2A, port 8001)
uv run lodestar "<posting URL>"       # terminal 2: ingest → analyze → review
uv run lodestar-mcp                   # the MCP server alone (for the MCP Inspector)
uv run lodestar discover [--dry-run] [--recheck]
uv run lodestar analyze [--budget N] [--dry-run]
uv run lodestar review [--limit N] | --dismiss JOB_ID [--reason TEXT]
uv run lodestar usage [--by run|day|model]
uv run lodestar ui                    # local web UI
uv run lodestar tailor JOB_ID         # resume for an approved job → data/resumes/
LODESTAR_DEMO=1 LODESTAR_DB=demo/demo.sqlite LODESTAR_PROFILE=demo/profile.yaml uv run lodestar ui
```

Config (`config/`, committed): `watchlist.yaml`, `filters.yaml` (pre-filter + ranking),
`prices.yaml`.

`.env`: `LODESTAR_FIT_MODELS` (comma-separated, tried in order, e.g.
`gemini-3.1-flash-lite,openai/gpt-5.4-nano,groq/openai/gpt-oss-120b`),
`GOOGLE_API_KEY`, `OPENAI_API_KEY`, `GROQ_API_KEY`; optional `LODESTAR_DB`,
`LODESTAR_PROFILE`, `LODESTAR_AGENT_PORT`, `LODESTAR_AGENT_URL`,
`LODESTAR_PAID_USD_PER_DAY` (batch paid-model cap, default 0). Budget: Gemini and Groq
on free tiers; OpenAI paid and limited, so fallback to it must stay controlled.
Placeholder model names are rejected at startup.

Logs: stderr, plus `data/logs/lodestar.log` (workflow) and
`data/logs/mcp_server.log` (MCP server subprocesses).

## Golden dataset and evaluation (`eval/`)

```text
eval/
  jobs/job_01.json ... job_05.json   # FULL posting content (saved JobPosting)
  expected.yaml                      # candidate's decision + reason
  runs/<label>/<job>.json            # agent results per run, with model and tokens
  save_posting.py                    # URL → jobs/job_NN.json (never overwrites without --force)
  make_expected.py                   # append stubs; --check validates
  run_fit.py                         # run the agent on golden jobs, report beside expected
  compare_runs.py                    # runs / groups side by side; --rescore = current scoring, no LLM
```

```bash
uv run python eval/run_fit.py --all --label <name>      # direct, uses LODESTAR_FIT_MODELS
uv run python eval/run_fit.py job_02 --models gemini-3.1-flash-lite
uv run python eval/run_fit.py --all --via-a2a           # through the running service
uv run python eval/run_fit.py --all --label prompt-v2c --repeat 3 --models gemini-3.1-flash-lite
uv run python eval/compare_runs.py "profile-v2*" "prompt-v2c-r*"   # quote patterns
uv run python eval/compare_runs.py "prompt-v2c-r*" --rescore
```

Results record the model and `prompt_version` (`PROMPT_VERSION` in `fit_agent/agent.py`).
Pin `--models` for eval runs so a fallback can't silently mix models. Missing analyses
can be added to an existing run folder by running only those jobs with the same label.

Save full content because postings get taken down. Record the reason because when
the agent disagrees, the reason shows which part is wrong. **Change one thing per
labeled run** (profile, prompt, scoring) so each change's effect is visible.

**LLM output is noisy.** With identical inputs, a job's score moves 0.05–0.20 between
runs. Judge a change on ≥3 runs per job (`--repeat 3`); differences under ~0.15 on one
job or ~0.05 on the five-job mean are noise. Temperature can't fix this: Gemini 3 is
tuned for its default of 1.0, and GPT-5 models don't accept other values.

v0.x findings (all on gemini-3.1-flash-lite):
- The profile (evidence ids, highlights) made judgments better grounded without
  measurably moving scores. v1's "mostly gaps" came from how matches were judged, not
  from the profile.
- Prompt v2 changed five rules at once; its "split multi-skill sentences" rule inflated
  requirements 30 → 41 per analysis and doubled gaps, so scores fell. v2b removed it.
  Lesson: one change per labeled run.
- v2c (hard_constraint field) + skill-only scoring + top_pick/possible/blocked, rescored:
  6 exact, 4 one step, 0 opposite out of 10 analyses.
- The model does not reliably follow "no inference from job titles" or "backend
  languages are related" (e.g. Perl/Go/Node stays a gap despite Java/C#/C++).

## Repo layout

```text
lodestar/
├── src/lodestar/
│   ├── schemas/        # shared Pydantic models
│   ├── db/             # SQLite layer
│   ├── ingest/         # fetch, classify, ATS adapters, extract, validate, ingest_url
│   ├── mcp_server/     # Lodestar MCP server → uv run lodestar-mcp; launch params; tool client
│   ├── fit_agent/      # FitService + to_a2a → uv run lodestar-agent (port 8001); A2A client
│   ├── workflow/       # LangGraph graph, analyze_one, CLI → uv run lodestar
│   ├── resume_agent/   # resume agent (ADK) + A2A service
│   ├── resume/         # resume checks, assembly, rendering
│   ├── app/            # service layer: models + functions any UI calls
│   ├── ui/             # Streamlit pages → uv run lodestar ui
│   ├── scoring.py, ranking.py, pricing.py, report.py, paths.py, quiet.py
├── config/             # watchlist.yaml, filters.yaml, prices.yaml
├── demo/               # fictional profile + postings, build_demo.py, demo.sqlite
├── eval/               # golden dataset and eval tools
├── reference/          # course examples (read-only)
├── data/               # gitignored: profile.yaml, lodestar.sqlite, logs/
└── tests/              # pytest; fake models, no live LLM calls
```

Services talk only over MCP and A2A, never by importing each other's internals
(shared `schemas/`, `db/`, and the MCP launch/client helpers excepted). The
workflow imports only the thin A2A client from `fit_agent`.

## Build history (v0)

1 schemas + profile → 2a ingestion pieces (pulled forward) → 2b golden dataset →
3 SQLite → 4 ingest_url + MCP server → 5 fit agent (local) → 7 LangGraph workflow
with a direct call → 8 terminal review → 6 A2A swap (done last, as a one-line swap
in the analyze node, so the workflow was proven before the transport changed).

## Build history (v1, first slice)

1 usage and cost tracking → 2 watchlist + board listing + dedupe + pre-filter
(`discover`) → 3 ranking + budgeted batch analysis + paid-spend guard (`analyze`) →
4 review of analyzed jobs + dismiss (`review`).

## Build history (UI)

A service layer (+ record_decision consolidation) → B Streamlit Phase 1 (see and decide) →
Phase 2 (Find / Analyze from the UI; CLI discover/analyze refactored onto the service
layer; company backfill for known jobs) → layout and Arrow fixes → overload retry.

## Portfolio plan

1 git + GitHub (done; ignored: data/, .env, eval/jobs/, eval/runs/, eval/expected.yaml,
reference/) → 2 v2 resume tailoring (done) → 3 demo mode + demo data + README (done; README
needs screenshots in docs/screenshots/) → 4 deploy the read-only demo on Render.

## v0.x: remaining

- `profile-v3`: API design highlight (where it was done), SAT apps "deployed on Render,
  used by family" (shipped to real users, small scale). Observability stays a real gap.
  Run `--repeat 3`, then tune `TOP_PICK_MIN` with `--rescore` (0.60 would match job_03).
- Later experiments: Gemini thinking level (Flash-Lite defaults to minimal), or a
  stronger model, for rule-following.
- Code check still open: flag direct/related matches with no evidence (hard constraints
  excepted) and evidence that doesn't support the requirement (e.g. a degree for location).

## v1 constraint: control LLM usage

v1 may discover far more jobs than v0 processes by hand. Running the fit agent on every
posting would waste tokens, raise cost and make rate limits more likely. So:

1. Deterministic pre-filter before the LLM: title, location/work mode, already-seen jobs,
   obvious mismatches. Log why each job was skipped.
2. Rank candidates before analysis; send only plausible jobs to the fit stage.
3. Analysis budget / queue: cap analyses per run/day; leave the rest queued.
4. Provider fallback: keep v0's Gemini → OpenAI → Groq per-run fallback for rate/size/outage.
5. Measure first: log calls, tokens, failures, provider/model and cost per call. Decide on
   a paid tier from measured workload, not assumed limits (limits vary by model, project
   and tier — RPM, TPM, RPD — and should be checked in the provider console).

```text
1000 discovered ─► deterministic filtering ($0) ─► ~120 ─► cheap ranking ─► ~25
                 ─► fit agent (LLM spend here) ─► ~8 ─► human review
```

The fit agent should spend reasoning on jobs that passed obvious checks; an LLM should
never be needed to learn that "Senior iOS Engineer — London" isn't relevant.

The first v1 slice implements this (see Discovery / Ranking / Usage above).

Possible further v1 work (not required for the loop): Serper search for companies not
on the watchlist; detecting the same job on different boards; LLM fallback for non-ATS
pages; marking jobs closed when they disappear from a board (`discoveries.last_seen_at`);
watchlist upkeep from discovery stats.

## Open decisions

- `TOP_PICK_MIN` and weights: provisional until v1 produces more real decisions.
- Model: `gemini-3.1-flash-lite` works as primary. Groq's free tier rejected a ~10K-token
  request against an 8K tokens/min limit for gpt-oss-120b (logged error); untested since
  tool results were compacted.
- Re-analysis: add an analyzed → queued transition, or keep analyses immutable?
- The LLM classification/extraction fallback for non-ATS pages.

## Roadmap (proposed, not confirmed)

- **v0.x — fit quality** (mostly done; see "v0.x: remaining").
- **v1 — finding jobs at volume:** first slice done (watchlist discovery, pre-filter,
  ranking, budgeted analysis, review, usage). Further items listed above, optional.
- **v2 — applying:** resume tailoring (first slice done). Later: resume parsing into
  `profile.draft.yaml`, `ready_to_apply → applied` tracking.
- **v3 — insight:** gap report, skill suggestions from gaps, agreement metrics;
  FastAPI + React over the service layer if the UI becomes a portfolio piece.

## Conventions

- Python 3.12, managed with `uv`
- Secrets in `.env` (never committed); provide `.env.example`
- Use `logging`, not `print` (tables go to stdout through a report logger)
- pytest for tests; no live LLM calls in tests
- Work on one build step at a time; explain designs before writing code and wait
  for approval; verify each step works before starting the next
