"""Turn a checked TailoredResume into the final resume: facts from the profile, text from the agent."""

from pydantic import BaseModel

from lodestar.schemas.profile import Contact, Profile
from lodestar.schemas.resume import TailoredResume


class ResumeRole(BaseModel):
    title: str
    organization: str
    dates: str
    highlights: list[str]


class ResumeProject(BaseModel):
    name: str
    url: str | None
    highlights: list[str]


class ResumeSchool(BaseModel):
    title: str
    institution: str
    year: int | None


class FinalResume(BaseModel):
    name: str
    contact: Contact | None
    headline: str
    summary: str
    skills: list[str]
    experience: list[ResumeRole]
    projects: list[ResumeProject]
    education: list[ResumeSchool]


def _dates(start: str, end: str | None) -> str:
    return f"{start[:4]} – {end[:4] if end else 'present'}"


def assemble(resume: TailoredResume, profile: Profile) -> FinalResume:
    experience = {e.id: e for e in profile.experience}
    projects = {p.id: p for p in profile.projects}
    chosen = sorted((e for e in resume.experience if e.entry_id in experience),
                    key=lambda e: experience[e.entry_id].start, reverse=True)  # newest first
    return FinalResume(
        name=profile.name,
        contact=profile.contact,
        headline=resume.headline,
        summary=resume.summary.text,
        skills=resume.skills,
        experience=[ResumeRole(title=experience[e.entry_id].title, organization=experience[e.entry_id].company,
                               dates=_dates(experience[e.entry_id].start, experience[e.entry_id].end),
                               highlights=[h.text for h in e.highlights]) for e in chosen],
        projects=[ResumeProject(name=projects[p.entry_id].name, url=projects[p.entry_id].url,
                                highlights=[h.text for h in p.highlights])
                  for p in resume.projects if p.entry_id in projects],
        education=[ResumeSchool(title=ed.title, institution=ed.institution, year=ed.year)
                   for ed in sorted(profile.education, key=lambda ed: ed.year or 0, reverse=True)],
    )
