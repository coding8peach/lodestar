"""Fit score and recommendation, computed in Python from the agent's per-requirement judgments.

LLM -> semantic judgment. Python -> arithmetic. Weights and thresholds are
starting values, to be tuned against the golden dataset, not by eye.
"""

from lodestar.schemas.fit import FitAnalysis, FitResult, RequirementMatch

MATCH_CREDIT = {"direct": 1.0, "related": 0.6, "gap": 0.0}
PRIORITY_WEIGHT = {"must_have": 2.0, "nice_to_have": 1.0}

# top_pick at or above this skill-fit score; everything else not blocked is "possible".
# Provisional: tune with `compare_runs.py --rescore` against real decisions.
TOP_PICK_MIN = 0.70


def unmet_hard_constraints(analysis: FitAnalysis) -> list[RequirementMatch]:
    """Must-have hard constraints (location, work authorization, ...) the candidate doesn't meet."""
    return [r for r in analysis.requirements
            if r.hard_constraint and r.priority == "must_have" and r.match_level == "gap"]


def score_analysis(analysis: FitAnalysis) -> FitResult:
    """Skill fit: weighted average of match credit over the non-hard-constraint
    requirements, weighted by priority, in [0, 1]; and a recommendation:

    - blocked:  an unmet must-have hard constraint; the candidate can't apply
    - top_pick: score >= TOP_PICK_MIN; apply first
    - possible: everything else; not a top pick, but not ruled out, whatever the score

    Hard constraints (location, work authorization, ...) are left out of the score:
    they decide whether a job is possible at all, not how well the skills fit, and
    counting a met one as full must-have credit would inflate every job whose location
    suits the candidate. They only decide "blocked".
    """
    skills = [r for r in analysis.requirements if not r.hard_constraint]
    if not skills:
        raise ValueError(f"job {analysis.job_id}: analysis has no skill requirements to score")
    total_weight = sum(PRIORITY_WEIGHT[r.priority] for r in skills)
    earned = sum(PRIORITY_WEIGHT[r.priority] * MATCH_CREDIT[r.match_level] for r in skills)
    score = round(earned / total_weight, 3)
    if unmet_hard_constraints(analysis):
        recommendation = "blocked"
    elif score >= TOP_PICK_MIN:
        recommendation = "top_pick"
    else:
        recommendation = "possible"
    return FitResult(job_id=analysis.job_id, analysis=analysis, overall_score=score, recommendation=recommendation)
