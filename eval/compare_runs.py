"""Compare eval runs side by side, one column per run or group of runs.

    uv run python eval/compare_runs.py profile-v1 profile-v2
    uv run python eval/compare_runs.py "profile-v2*" "prompt-v2-r*"     (quote the *)
    uv run python eval/compare_runs.py "prompt-v2c-r*" --rescore         (current scoring.py, no LLM)

Each argument is a run folder under eval/runs/, or a pattern matching several
(a group, e.g. the -r1..-r3 folders from `run_fit.py --repeat 3`). Per job and
group: mean score, [min-max] range, and each run's recommendation as a letter
(T top pick, P possible, B blocked; s / p / w for older runs). Per group: match-level counts per run,
agreement with your decisions across all its runs, mean score, tokens.
Decisions come from the current expected.yaml.

--rescore recomputes every score and recommendation from the saved analyses with the
current scoring.py (weights, thresholds, hard-constraint cap), so scoring changes can be
compared without new LLM runs. Runs saved before prompt v2c have no hard-constraint marks,
so the cap can't apply to them; the summary says so.

Agreement maps top_pick / possible / blocked to apply / maybe / skip (runs saved before
the change used strong_fit / possible_fit / weak_fit, mapped the same way; --rescore
recomputes them with the current rules). "exact" = same, "one step" = adjacent,
"opposite" = top pick vs skip or blocked vs apply.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from statistics import mean

import yaml

from lodestar.report import report_logger
from lodestar.schemas.fit import FitAnalysis
from lodestar.scoring import score_analysis

EVAL_DIR = Path(__file__).resolve().parent
LEVEL = {"top_pick": 2, "possible": 1, "blocked": 0,
         "strong_fit": 2, "possible_fit": 1, "weak_fit": 0}  # older runs
DECISION_LEVEL = {"apply": 2, "maybe": 1, "skip": 0}
LETTER = {"top_pick": "T", "possible": "P", "blocked": "B",
          "strong_fit": "s", "possible_fit": "p", "weak_fit": "w"}  # lower case: older runs


def load_run(folder: Path) -> dict[str, dict]:
    """job name -> saved result file content."""
    return {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted(folder.glob("job_*.json"))}


def resolve(pattern: str, runs_dir: Path) -> list[Path]:
    """Run folders for one argument: an exact folder name, or a glob pattern."""
    exact = runs_dir / pattern
    if exact.is_dir():
        return [exact]
    matches = sorted(p for p in runs_dir.glob(pattern) if p.is_dir())
    if not matches:
        raise FileNotFoundError(f"no run folder matches {pattern!r} in {runs_dir}")
    return matches


def agreement(recommendation: str, decision: str | None) -> str | None:
    if decision not in DECISION_LEVEL:
        return None
    return {0: "exact", 1: "one step", 2: "opposite"}[abs(LEVEL[recommendation] - DECISION_LEVEL[decision])]


def _rescored(saved: dict) -> dict:
    analysis = FitAnalysis.model_validate(saved["result"]["analysis"])
    return {**saved, "result": score_analysis(analysis).model_dump(mode="json")}


def compare(groups: list[str], runs_dir: Path, expected: dict, rescore: bool = False) -> dict:
    """Everything the report prints, as data (so it can be tested)."""
    folders = {g: [f.name for f in resolve(g, runs_dir)] for g in groups}
    owner: dict[str, str] = {}
    for g in groups:
        for name in folders[g]:
            if name in owner:
                raise ValueError(f"run {name!r} matches both {owner[name]!r} and {g!r}; use narrower patterns")
            owner[name] = g
    runs = {g: [load_run(runs_dir / name) for name in folders[g]] for g in groups}
    # Whether the agent marked hard constraints; checked on the saved data, before any rescoring.
    hard_marks = {g: any("hard_constraint" in r for run in runs[g] for saved in run.values()
                         for r in saved["result"]["analysis"]["requirements"]) for g in groups}
    if rescore:
        runs = {g: [{job: _rescored(saved) for job, saved in run.items()} for run in rs] for g, rs in runs.items()}
    jobs = sorted({job for g in groups for run in runs[g] for job in run})
    decision = {job: (expected.get(job) or {}).get("human_decision") for job in jobs}

    rows = []
    for job in jobs:
        cells = {}
        for g in groups:
            results = [run[job]["result"] for run in runs[g] if job in run]
            if not results:
                continue
            scores = [r["overall_score"] for r in results]
            cells[g] = {
                "mean": round(mean(scores), 3),
                "min": min(scores),
                "max": max(scores),
                "recommendations": [r["recommendation"] for r in results],
            }
        rows.append({"job": job, "cells": cells, "decision": decision[job]})

    summaries = {}
    for g in groups:
        levels = {"direct": 0, "related": 0, "gap": 0}
        agree = {"exact": 0, "one step": 0, "opposite": 0}
        tokens, models, prompts, requirements, scores = 0, set(), set(), 0, []
        for run in runs[g]:
            for job, saved in run.items():
                result = saved["result"]
                requirements += len(result["analysis"]["requirements"])
                for r in result["analysis"]["requirements"]:
                    levels[r["match_level"]] += 1
                a = agreement(result["recommendation"], decision.get(job))
                if a:
                    agree[a] += 1
                scores.append(result["overall_score"])
                tokens += saved.get("tokens") or 0
                models.add(saved.get("model") or "?")
                prompts.add(saved.get("prompt_version") or "v1")  # runs before versioning used v1
        n = sum(len(run) for run in runs[g]) or 1  # analyses, not run folders: runs can be incomplete
        summaries[g] = {
            "runs": folders[g],
            "analyses": sum(len(run) for run in runs[g]),
            "requirements_per_analysis": round(requirements / n, 1),
            "levels_per_analysis": {k: round(v / n, 1) for k, v in levels.items()},
            "agreement": agree,
            "mean_score": round(mean(scores), 3) if scores else None,
            "tokens_per_analysis": round(tokens / n),
            "models": sorted(models),
            "prompts": sorted(prompts),
            "hard_constraint_data": hard_marks[g],
        }
    return {"groups": groups, "rows": rows, "summaries": summaries, "rescored": rescore}


def _cell(cell: dict | None) -> str:
    if cell is None:
        return "-"
    letters = "".join(LETTER[r] for r in cell["recommendations"])
    if cell["min"] == cell["max"]:
        return f"{cell['mean']:.2f} {letters}"
    return f"{cell['mean']:.2f} [{cell['min']:.2f}-{cell['max']:.2f}] {letters}"


def print_report(data: dict) -> None:
    out = report_logger()
    groups = data["groups"]
    cells = {row["job"]: [_cell(row["cells"].get(g)) for g in groups] for row in data["rows"]}
    width = max([len(g) for g in groups] + [len(c) for cs in cells.values() for c in cs]) + 2
    out.info("")
    out.info("job     " + "".join(f"{g:<{width}}" for g in groups) + "you")
    out.info("-" * (8 + width * len(groups) + 6))
    for row in data["rows"]:
        out.info(f"{row['job']:<8}" + "".join(f"{c:<{width}}" for c in cells[row["job"]]) + (row["decision"] or "?"))
    out.info("score = mean [min-max]; letters = each run's recommendation "
             "(T top pick, P possible, B blocked; s/p/w = strong/possible/weak in older runs)")
    if data["rescored"]:
        out.info("rescored with the current scoring.py")
    out.info("")
    for g in groups:
        s = data["summaries"][g]
        lv, ag = s["levels_per_analysis"], s["agreement"]
        runs_text = f"{len(s['runs'])} run{'s' if len(s['runs']) > 1 else ''}: {', '.join(s['runs'])}"
        out.info(f"{g}  ({s['analyses']} analyses in {runs_text})")
        out.info(f"  per analysis: {s['requirements_per_analysis']} requirements "
                 f"(direct {lv['direct']}, related {lv['related']}, gap {lv['gap']}), "
                 f"tokens {s['tokens_per_analysis']}")
        out.info(f"  vs you, all runs: {ag['exact']} exact, {ag['one step']} one step, {ag['opposite']} opposite; "
                 f"mean score {s['mean_score']}; model {', '.join(s['models'])}; prompt {', '.join(s['prompts'])}")
        if data["rescored"] and not s["hard_constraint_data"]:
            out.info("  no hard-constraint marks in these runs (before prompt v2c): the cap can't apply")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("groups", nargs="+", help='run folder names or patterns under eval/runs/, e.g. "prompt-v2-r*"')
    parser.add_argument("--rescore", action="store_true", help="recompute scores with the current scoring.py")
    args = parser.parse_args()
    expected = yaml.safe_load((EVAL_DIR / "expected.yaml").read_text(encoding="utf-8")) or {}
    try:
        data = compare(args.groups, EVAL_DIR / "runs", expected, rescore=args.rescore)
    except (FileNotFoundError, ValueError) as e:
        parser.error(str(e))
    print_report(data)
    return 0


if __name__ == "__main__":
    sys.exit(main())
