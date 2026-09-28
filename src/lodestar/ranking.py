"""Rank queued jobs before any LLM call: which ones are worth the analysis budget first.

Deterministic and explainable. Three parts:
  1. Near-duplicate groups: same company + same title without seniority words
     ("Senior ... Backend" and "Staff ... Backend" are one role). Once any job in a group
     is analyzed, the rest are skipped; within the queue, only the best of a group is picked.
  2. Relevance score: profile skills found in the title (x3) and description (capped),
     weighted by depth (production 3, project 2, course 1); frontend-heavy terms subtract.
  3. A per-company cap per run.
The score only orders the queue. It is never a decision; the fit agent judges.
"""

import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from lodestar.ingest.prefilter import FILTERS_FILE
from lodestar.schemas.job import JobPosting
from lodestar.schemas.profile import Profile

DEPTH_WEIGHT = {"production": 3, "project": 2, "course": 1}
TITLE_MULTIPLIER = 3
DESCRIPTION_CAP = 3      # occurrences per keyword counted from the description
PENALTY_TITLE = 6        # a penalized term in the title
PENALTY_DESCRIPTION = 1  # per description occurrence, capped at DESCRIPTION_CAP

_SENIORITY = re.compile(r"\b(sr\.?\s*staff|senior\s+staff|staff/sr\.?\s*staff|staff|senior|sr\.?|lead)\b", re.I)


def normalize_title(title: str) -> str:
    """Title without seniority words or punctuation, for spotting near-duplicates."""
    t = _SENIORITY.sub(" ", title.lower())
    t = re.sub(r"[^\w+#]+", " ", t)
    return " ".join(t.split())


def group_key(job: JobPosting) -> tuple[str, str]:
    return ((job.company or "").lower().strip(), normalize_title(job.title))


def _term(word: str) -> re.Pattern[str]:
    # Word-ish boundaries that also work for terms like "c++" and "c#".
    return re.compile(r"(?<![\w+#])" + re.escape(word.lower()) + r"(?![\w+#])", re.I)


def profile_keywords(profile: Profile) -> dict[str, int]:
    """keyword -> weight from the profile's skills. "C/C++" -> c, c++ (1-letter terms dropped)."""
    keywords: dict[str, int] = {}
    for skill in profile.skills:
        for part in re.split(r"\s*/\s*", skill.name):
            part = part.strip().lower()
            if len(part) > 1:
                keywords[part] = max(keywords.get(part, 0), DEPTH_WEIGHT[skill.depth])
    return keywords


@dataclass
class Ranked:
    job: JobPosting
    score: int
    matched: list[str]
    group: tuple[str, str]
    similar: list[JobPosting] = field(default_factory=list)  # other queued jobs in the same group


@dataclass
class RankConfig:
    max_per_company_per_run: int = 2
    penalize: tuple[str, ...] = ()


def load_rank_config(path: Path = FILTERS_FILE) -> RankConfig:
    cfg = (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("ranking") or {}
    return RankConfig(max_per_company_per_run=int(cfg.get("max_per_company_per_run", 2)),
                      penalize=tuple(cfg.get("penalize") or ()))


def score_job(job: JobPosting, keywords: dict[str, int], penalize: tuple[str, ...]) -> tuple[int, list[str]]:
    score, matched = 0, []
    penalized = {w.lower() for w in penalize}
    for word, weight in keywords.items():
        if word in penalized:  # e.g. React is a profile skill but a frontend-heavy signal: penalty only
            continue
        pattern = _term(word)
        in_title = bool(pattern.search(job.title))
        in_body = min(len(pattern.findall(job.description)), DESCRIPTION_CAP)
        if in_title or in_body:
            score += weight * (TITLE_MULTIPLIER * in_title + in_body)
            matched.append(word)
    for word in penalize:
        pattern = _term(word)
        penalty = (PENALTY_TITLE if pattern.search(job.title) else 0) + \
            PENALTY_DESCRIPTION * min(len(pattern.findall(job.description)), DESCRIPTION_CAP)
        if penalty:
            score -= penalty
            matched.append(f"-{word}")
    return score, matched


def rank_queue(
    queued: list[JobPosting], analyzed: list[JobPosting], profile: Profile, config: RankConfig
) -> tuple[list[Ranked], list[tuple[JobPosting, JobPosting]]]:
    """(ranked groups best first, [(skipped job, analyzed job it resembles)])."""
    keywords = profile_keywords(profile)
    done = {group_key(j): j for j in analyzed}
    already, groups = [], {}
    for job in queued:
        key = group_key(job)
        if key in done:
            already.append((job, done[key]))
            continue
        score, matched = score_job(job, keywords, config.penalize)
        candidate = Ranked(job, score, matched, key)
        best = groups.get(key)
        if best is None:
            groups[key] = candidate
        elif (candidate.score, str(candidate.job.date_posted)) > (best.score, str(best.job.date_posted)):
            candidate.similar = [best.job, *best.similar]
            groups[key] = candidate
        else:
            best.similar.append(job)
    # best score first; newer posting first among equal scores
    ranked = sorted(groups.values(),
                    key=lambda r: (-r.score, -(r.job.date_posted.toordinal() if r.job.date_posted else 0)))
    return ranked, already


def pick(ranked: list[Ranked], budget: int, config: RankConfig) -> list[Ranked]:
    """Top `budget` groups, at most max_per_company_per_run from one company."""
    picked, per_company = [], {}
    for r in ranked:
        if len(picked) >= budget:
            break
        company = r.group[0]
        if per_company.get(company, 0) >= config.max_per_company_per_run:
            continue
        per_company[company] = per_company.get(company, 0) + 1
        picked.append(r)
    return picked


def load_queue(conn: sqlite3.Connection) -> tuple[list[JobPosting], list[JobPosting]]:
    """(queued jobs, jobs that already have an analysis)."""
    from lodestar.db import get_job

    queued = [get_job(conn, r[0]) for r in conn.execute("SELECT id FROM jobs WHERE status = 'queued'")]
    analyzed = [get_job(conn, r[0]) for r in conn.execute(
        "SELECT DISTINCT job_id FROM fit_results")]
    return queued, analyzed
