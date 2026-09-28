"""Save one real posting into the golden dataset.

    uv run python eval/save_posting.py <url> job_01

Writes eval/jobs/job_01.json. Existing files are never overwritten unless you
pass --force: the golden dataset is a frozen baseline.
"""

import argparse
import logging
import re
import sys
from pathlib import Path

import httpx

from lodestar.ingest import ExtractionError, build_posting, validate_job

log = logging.getLogger("save_posting")
JOBS_DIR = Path(__file__).resolve().parent / "jobs"


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("url")
    parser.add_argument("name", help="dataset name, e.g. job_01")
    parser.add_argument("--source", default="pasted_url", choices=["pasted_url", "greenhouse", "lever", "ashby"])
    parser.add_argument("--force", action="store_true", help="overwrite an existing file")
    args = parser.parse_args()

    if not re.fullmatch(r"job_\d{2}", args.name):
        log.error("name must look like job_01")
        return 2
    out = JOBS_DIR / f"{args.name}.json"
    if out.exists() and not args.force:
        log.error("%s already exists (use --force to overwrite)", out)
        return 1

    try:
        job = validate_job(build_posting(args.url, source=args.source))
    except ExtractionError as e:
        log.error("extraction failed: %s", e)
        return 1
    except httpx.HTTPError as e:
        log.error("request failed: %s", e)
        return 1

    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    out.write_text(job.model_dump_json(indent=2) + "\n", encoding="utf-8")
    log.info("saved %s", out)
    log.info("  %s | %s | %s | work_mode=%s | salary=%s", job.title, job.company, job.location, job.work_mode, job.salary_text)
    log.info("  classified_by=%s | description: %d chars", job.classified_by, len(job.description))
    return 0


if __name__ == "__main__":
    sys.exit(main())
