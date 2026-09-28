"""Maintain eval/expected.yaml, the candidate's own decisions for the golden dataset.

    uv run python eval/make_expected.py           # append stubs for jobs not yet in expected.yaml
    uv run python eval/make_expected.py --check   # validate entries against eval/jobs/

Stubs are appended as text, so existing entries and comments are never rewritten.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from urllib.parse import urlsplit

import yaml

log = logging.getLogger("make_expected")

EVAL_DIR = Path(__file__).resolve().parent
DECISIONS = ("apply", "maybe", "skip")
HEADER = (
    "# Golden dataset: your own decision for each saved posting, written before seeing any agent output.\n"
    "# human_decision: apply / maybe / skip. reason: what drove the decision.\n"
)


def load_jobs(jobs_dir: Path) -> dict[str, dict]:
    """job name (file stem) -> parsed JSON."""
    return {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted(jobs_dir.glob("job_*.json"))}


def load_expected(expected_path: Path) -> dict:
    if not expected_path.exists():
        return {}
    data = yaml.safe_load(expected_path.read_text(encoding="utf-8"))
    return data or {}


def _board(url: str | None) -> str | None:
    """Board slug from a hosted ATS URL, e.g. 'close' from jobs.ashbyhq.com/close/<id>."""
    if not url:
        return None
    segments = [s for s in urlsplit(url).path.split("/") if s]
    return segments[0] if segments else None


def _summary(job: dict) -> str:
    company = job.get("company")
    if not company:
        board = _board(job.get("url"))
        company = f"{board} (board)" if board else None
    parts = [job.get("title"), company, job.get("location")]
    return " | ".join(" ".join(str(p).split()) for p in parts if p)


def stub(name: str, job: dict) -> str:
    url_line = f"  # {job['url']}\n" if job.get("url") else ""
    return (
        f"\n{name}:  # {_summary(job)}\n"
        f"{url_line}"
        f"  human_decision:   # {' / '.join(DECISIONS)}\n"
        f'  reason: ""\n'
    )


def add_stubs(jobs_dir: Path, expected_path: Path) -> list[str]:
    """Append a stub for every job missing from expected.yaml. Returns the names added."""
    jobs = load_jobs(jobs_dir)
    existing = load_expected(expected_path)
    missing = [name for name in jobs if name not in existing]
    if not missing:
        return []
    text = expected_path.read_text(encoding="utf-8") if expected_path.exists() else HEADER
    if not text.endswith("\n"):
        text += "\n"
    text += "".join(stub(name, jobs[name]) for name in missing)
    expected_path.write_text(text, encoding="utf-8")
    return missing


def check(jobs_dir: Path, expected_path: Path) -> list[str]:
    """Return a list of problems; empty means the dataset is consistent."""
    jobs = load_jobs(jobs_dir)
    expected = load_expected(expected_path)
    problems: list[str] = []
    for name in jobs:
        if name not in expected:
            problems.append(f"{name}: job file has no entry in expected.yaml")
    for name, entry in expected.items():
        if name not in jobs:
            problems.append(f"{name}: entry has no job file in {jobs_dir.name}/")
        if not isinstance(entry, dict):
            problems.append(f"{name}: entry should have human_decision and reason")
            continue
        if entry.get("human_decision") not in DECISIONS:
            problems.append(f"{name}: human_decision must be one of {', '.join(DECISIONS)}")
        reason = entry.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            problems.append(f"{name}: reason is empty")
    return problems


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="validate instead of adding stubs")
    args = parser.parse_args()
    jobs_dir, expected_path = EVAL_DIR / "jobs", EVAL_DIR / "expected.yaml"

    if args.check:
        problems = check(jobs_dir, expected_path)
        for p in problems:
            log.error(p)
        if not problems:
            log.info("expected.yaml OK (%d entries)", len(load_expected(expected_path)))
        return 1 if problems else 0

    added = add_stubs(jobs_dir, expected_path)
    if added:
        log.info("added stubs for %s to %s", ", ".join(added), expected_path)
    else:
        log.info("nothing to add: every job already has an entry")
    return 0


if __name__ == "__main__":
    sys.exit(main())
