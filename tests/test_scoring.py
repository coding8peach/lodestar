import pytest

from lodestar.schemas import FitAnalysis, RequirementMatch
from lodestar.scoring import score_analysis


def req(priority, level):
    return RequirementMatch(requirement=f"{priority}-{level}", priority=priority, evidence=[],
                            match_level=level, explanation="")


def analysis(*reqs):
    return FitAnalysis(job_id="job_x", requirements=list(reqs), summary="")


def test_all_direct_is_strong():
    result = score_analysis(analysis(req("must_have", "direct"), req("nice_to_have", "direct")))
    assert result.overall_score == 1.0 and result.recommendation == "top_pick"


def test_weights():
    # must direct (2*1.0) + must related (2*0.6) + nice gap (1*0) = 3.2 / 5
    result = score_analysis(analysis(req("must_have", "direct"), req("must_have", "related"),
                                     req("nice_to_have", "gap")))
    assert result.overall_score == 0.64 and result.recommendation == "possible"


def test_missing_nice_to_have_does_not_sink():
    reqs = [req("must_have", "direct")] * 3 + [req("nice_to_have", "gap")] * 2
    assert score_analysis(analysis(*reqs)).recommendation == "top_pick"  # 6/8 = 0.75


def test_single_must_have_gap_does_not_sink():
    reqs = [req("must_have", "direct")] * 4 + [req("must_have", "gap")]
    assert score_analysis(analysis(*reqs)).recommendation == "top_pick"  # 8/10


def test_low_score_is_still_possible():
    # not a top pick, but not ruled out: only a hard constraint rules a job out
    reqs = [req("must_have", "gap")] * 3 + [req("must_have", "related")]
    result = score_analysis(analysis(*reqs))
    assert result.overall_score == 0.15 and result.recommendation == "possible"


def test_no_requirements_rejected():
    with pytest.raises(ValueError):
        score_analysis(analysis())


def hard(level, priority="must_have"):
    return RequirementMatch(requirement="Eligible to work in Canada", priority=priority, evidence=[],
                            match_level=level, explanation="", hard_constraint=True)


def test_unmet_must_have_hard_constraint_caps_at_weak():
    result = score_analysis(analysis(*[req("must_have", "direct")] * 4, hard("gap")))
    assert result.overall_score == 1.0          # skill fit only: the hard constraint isn't in the score
    assert result.recommendation == "blocked"   # but the blocker rules the job out


def test_met_hard_constraint_does_not_block():
    result = score_analysis(analysis(*[req("must_have", "direct")] * 4, hard("direct")))
    assert result.recommendation == "top_pick"


def test_nice_to_have_hard_constraint_does_not_block():
    result = score_analysis(analysis(*[req("must_have", "direct")] * 4, hard("gap", "nice_to_have")))
    assert result.recommendation == "top_pick"


def test_old_analyses_default_to_no_hard_constraints():
    assert RequirementMatch(requirement="x", priority="must_have", evidence=[], match_level="gap",
                            explanation="").hard_constraint is False


def test_met_hard_constraint_does_not_raise_the_score():
    # 2 related must-haves score 0.6; a met location requirement must not lift that
    result = score_analysis(analysis(req("must_have", "related"), req("must_have", "related"), hard("direct")))
    assert result.overall_score == 0.6 and result.recommendation == "possible"


def test_only_hard_constraints_cannot_be_scored():
    with pytest.raises(ValueError, match="no skill requirements"):
        score_analysis(analysis(hard("direct")))


def test_top_pick_cutoff():
    # 0.70 exactly is a top pick; just below is possible
    at = [req("must_have", "direct")] * 7 + [req("must_have", "gap")] * 3                     # 0.70
    below = [req("must_have", "direct")] * 6 + [req("must_have", "related"), req("must_have", "gap"),
                                                 req("must_have", "gap"), req("must_have", "gap")]  # 0.66
    assert score_analysis(analysis(*at)).recommendation == "top_pick"
    assert score_analysis(analysis(*below)).recommendation == "possible"
