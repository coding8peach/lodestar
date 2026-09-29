"""Resume schemas. The resume agent returns TailoredResume: it selects, orders and rewords,
citing profile entries for every line. Python fills in all facts (names, titles, dates,
degrees, contact) from the profile and checks that nothing is invented."""

from pydantic import BaseModel

from lodestar.schemas.fit import LlmCall


class ResumeLine(BaseModel):
    text: str
    sources: list[str]           # profile entry ids this line is based on


class ResumeEntry(BaseModel):
    entry_id: str                # exp-... or proj-...; the rest of the entry comes from the profile
    highlights: list[ResumeLine] = []


class TailoredResume(BaseModel):
    job_id: str
    headline: str
    summary: ResumeLine
    skills: list[str]            # chosen and ordered; only the profile's skills
    experience: list[ResumeEntry]
    projects: list[ResumeEntry] = []
    notes: list[str] = []        # for the candidate: requirements not claimed, and why


class ResumeAgentReply(BaseModel):  # what the resume agent service returns over A2A
    resume: TailoredResume
    model: str
    skipped: list[str] = []
    tokens: int | None = None
    calls: list[LlmCall] = []
