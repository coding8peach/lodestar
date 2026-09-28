"""Small types shared by the profile and the job posting."""

from typing import Literal

Seniority = Literal["intern", "junior", "mid", "senior", "staff", "principal", "manager"]
WorkMode = Literal["remote", "hybrid", "onsite"]
SkillDepth = Literal["production", "project", "course"]
