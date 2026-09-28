"""The candidate profile: everything about the candidate, for the fit agent.

A resume is a curated selection for employers; the profile is the full record.
It is hand-written as YAML in v0 and loaded with `load_profile`.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from lodestar.schemas.common import Seniority, SkillDepth, WorkMode

# "2010" or "2010-03"
YEAR_MONTH = r"^\d{4}(-(0[1-9]|1[0-2]))?$"


class _Strict(BaseModel):
    """Base for profile models: unknown keys fail, so YAML typos surface at load time."""

    model_config = ConfigDict(extra="forbid")


class TargetPreferences(_Strict):
    roles: list[str]
    seniority: list[Seniority]
    locations: list[str]
    work_modes: list[WorkMode]
    frontend_preference: Literal["frontend_heavy_ok", "light_frontend_ok", "backend_only"]
    notes: str | None = None


class Experience(_Strict):
    id: str
    company: str
    title: str
    start: str = Field(pattern=YEAR_MONTH)
    end: str | None = Field(default=None, pattern=YEAR_MONTH)  # None = current
    highlights: list[str] = []
    technologies: list[str] = []


class Project(_Strict):
    id: str
    name: str
    description: str
    url: str | None = None
    highlights: list[str] = []
    technologies: list[str] = []


class Education(_Strict):
    id: str
    kind: Literal["degree", "certificate", "course"]
    institution: str
    title: str
    year: int | None = None


class Skill(_Strict):
    """One entry per skill: the deepest level ever reached, and the latest year used at any level."""

    name: str
    depth: SkillDepth
    last_used: int = Field(ge=1970)
    evidence: list[str] = []  # ids of Experience / Project / Education entries

    @model_validator(mode="after")
    def _not_in_future(self) -> Skill:
        if self.last_used > date.today().year:
            raise ValueError(f"skill {self.name!r}: last_used {self.last_used} is in the future")
        return self


class SpokenLanguage(_Strict):
    name: str
    proficiency: Literal["native", "fluent", "professional", "conversational", "basic"]


class Profile(_Strict):
    name: str
    summary: str
    targets: TargetPreferences
    experience: list[Experience] = []
    projects: list[Project] = []
    education: list[Education] = []
    skills: list[Skill] = []
    languages: list[SpokenLanguage] = []

    @model_validator(mode="after")
    def _check_references(self) -> Profile:
        ids = [e.id for e in self.experience] + [p.id for p in self.projects] + [e.id for e in self.education]
        duplicates = sorted({i for i in ids if ids.count(i) > 1})
        if duplicates:
            raise ValueError(f"duplicate ids: {duplicates}")

        names = [s.name.casefold() for s in self.skills]
        dup_skills = sorted({n for n in names if names.count(n) > 1})
        if dup_skills:
            raise ValueError(f"skills listed more than once (use one entry per skill): {dup_skills}")

        known = set(ids)
        for skill in self.skills:
            missing = [ref for ref in skill.evidence if ref not in known]
            if missing:
                raise ValueError(f"skill {skill.name!r}: unknown evidence ids {missing}")
        return self

    def evidence_for(self, skill: Skill) -> list[Experience | Project | Education]:
        """Resolve a skill's evidence ids to the entries they point at."""
        by_id = {x.id: x for x in [*self.experience, *self.projects, *self.education]}
        return [by_id[ref] for ref in skill.evidence]


def load_profile(path: str | Path) -> Profile:
    """Read a profile YAML file and validate it."""
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return Profile.model_validate(data)
