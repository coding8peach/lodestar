"""The fit table shown in the terminal, shared by run_fit.py, the workflow and (step 8) the review."""

import logging
import sys
from collections.abc import Iterable

from lodestar.schemas.fit import FitResult
from lodestar.scoring import unmet_hard_constraints
from lodestar.schemas.job import JobPosting

WIDTH = 100


def fit_table(
    job: JobPosting,
    result: FitResult,
    model: str | None = None,
    *,
    heading: str = "",
    tokens: int | None = None,
    skipped: Iterable[str] = (),
    expected: dict | None = None,
    unknown_evidence: Iterable[tuple[str, str]] = (),
) -> list[str]:
    """Lines of text: job header, one row per requirement, summary and recommendation."""
    lines = ["", "=" * WIDTH]
    prefix = f"{heading}  " if heading else ""
    lines.append(f"{prefix}{job.title} | {job.company or '?'} | {job.location or '?'}")
    info = f"model: {model or '?'}"
    if tokens:
        info += f"   tokens: {tokens}"
    lines.append(info)
    lines += [f"skipped: {s}" for s in skipped]
    lines.append("-" * WIDTH)
    for m in result.analysis.requirements:
        pri = ("MUST" if m.priority == "must_have" else "nice") + ("!" if m.hard_constraint else "")
        lines.append(f"{m.match_level:<7} {pri:<5} {m.requirement}")
        lines.append(f"              {m.explanation}  [{', '.join(m.evidence) or '-'}]")
    lines.append("-" * WIDTH)
    if any(m.hard_constraint for m in result.analysis.requirements):
        lines.append("(! = hard constraint)")
    lines.append(f"summary: {result.analysis.summary}")
    lines.append(f"agent:   {result.recommendation} (score {result.overall_score:.2f})")
    blockers = unmet_hard_constraints(result.analysis)
    if blockers:
        lines.append("         blocked: hard constraint not met: " + "; ".join(r.requirement for r in blockers))
    if expected is not None:
        lines.append(f"you:     {expected.get('human_decision', '?')} | {expected.get('reason', '')}")
    bad = list(unknown_evidence)
    if bad:
        lines.append("WARNING: evidence ids not in the profile: " + ", ".join(f"{i} ({r})" for r, i in bad))
    return lines


class _CurrentStdout(logging.StreamHandler):
    """Writes to whatever sys.stdout is at the time (it can be swapped, e.g. by test capture)."""

    def emit(self, record: logging.LogRecord) -> None:
        self.stream = sys.stdout
        super().emit(record)


def report_logger(name: str = "lodestar.report") -> logging.Logger:
    """A logger that writes plain lines to stdout, for the tables users read.
    (CLAUDE.md: logging, not print. Diagnostic logs keep going to stderr.)"""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = _CurrentStdout(sys.stdout)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger
