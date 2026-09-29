"""Python's checks on a TailoredResume: the agent may select, order and reword, never invent.

1. Every entry id and every cited source exists in the profile (experience in experience,
   projects in projects).
2. Every skill is one of the profile's skills.
3. Every number in a line appears in the text of a cited source, except a number of years
   no larger than the whole career (e.g. "15+ years of backend experience").
4. Impact words ("high-scale", "proven", "expert", ...) appear only if a cited source, or the
   profile's own summary, uses them.
5. No placeholders (TODO, TBD, <...>) in the agent's text.
Placeholders in the profile's own facts can't be fixed by a retry: `profile_placeholders`
finds them so the caller can point to the entry to fix.
Required entries (all experience since REQUIRED_SINCE) are added by Python if the agent
left them out, with their original highlights.
"""

import re
from datetime import date

from lodestar.schemas.profile import Education, Experience, Profile, Project
from lodestar.schemas.resume import ResumeEntry, ResumeLine, TailoredResume

REQUIRED_SINCE = 2006
_NUMBER = re.compile(r"\d+(?:[.,]\d+)?")
_YEARS_CLAIM = re.compile(r"(\d+)\+?\s*(?:years|yrs)", re.I)
_PLACEHOLDER = re.compile(r"\bTODO\b|\bTBD\b|<[^<>]{1,40}>", re.I)

# Inflation the number check can't see. Allowed only when a cited source (or the profile's own
# summary) uses the word; stems match word forms ("specializ" -> specializing, specialized).
IMPACT_WORDS = (
    "high-scale", "large-scale", "at scale", "massive", "millions", "billions", "proven", "track record",
    "world-class", "robust", "expert", "expertise", "specializ", "seasoned", "extensive", "cutting-edge",
    "industry-leading", "spearheaded",
)


class ResumeRejected(ValueError):
    """The draft doesn't hold up against the profile; the message lists what's wrong."""


def _entries(profile: Profile) -> dict[str, Experience | Project | Education]:
    return {x.id: x for x in [*profile.experience, *profile.projects, *profile.education]}


def _source_text(entry: Experience | Project | Education) -> str:
    return " ".join(str(v) for v in entry.model_dump().values() if v)


def career_years(profile: Profile, today: date | None = None) -> int:
    starts = [int(e.start[:4]) for e in profile.experience]
    return ((today or date.today()).year - min(starts)) if starts else 0


def _impact_ok(line: ResumeLine, entries: dict, profile_summary: str) -> list[str]:
    source = (" ".join(_source_text(entries[s]) for s in line.sources if s in entries) + " "
              + profile_summary).lower()
    text = line.text.lower()
    unsupported = [w for w in IMPACT_WORDS if w in text and w not in source]
    if unsupported:
        return [f'"{line.text[:60]}": claim word(s) {", ".join(unsupported)} not supported by its sources; '
                "drop them or use the sources' own words"]
    return []


def _numbers_ok(line: ResumeLine, entries: dict, max_years: int) -> list[str]:
    source = " ".join(_source_text(entries[s]) for s in line.sources if s in entries)
    source_numbers = set(_NUMBER.findall(source))
    year_claims = {m.group(1) for m in _YEARS_CLAIM.finditer(line.text) if int(m.group(1)) <= max_years}
    bad = [n for n in _NUMBER.findall(line.text) if n not in source_numbers and n not in year_claims]
    return [f'"{line.text[:60]}": number(s) {", ".join(bad)} not in its sources'] if bad else []


def validate_resume(resume: TailoredResume, profile: Profile, job_id: str) -> list[str]:
    """All problems found; empty means the draft holds up."""
    problems: list[str] = []
    entries = _entries(profile)
    experience_ids = {e.id for e in profile.experience}
    project_ids = {p.id for p in profile.projects}
    skill_names = {s.name.lower() for s in profile.skills}
    max_years = career_years(profile)

    if resume.job_id != job_id:
        problems.append(f"job_id {resume.job_id!r} is not {job_id!r}")
    for kind, items, allowed in (("experience", resume.experience, experience_ids),
                                 ("projects", resume.projects, project_ids)):
        for item in items:
            if item.entry_id not in allowed:
                problems.append(f"{kind}: {item.entry_id!r} is not a {kind} entry in the profile")
    lines = [resume.summary] + [h for item in [*resume.experience, *resume.projects] for h in item.highlights]
    for line in lines:
        unknown = [s for s in line.sources if s not in entries]
        if not line.sources:
            problems.append(f'"{line.text[:60]}": cites no profile entry')
        if unknown:
            problems.append(f'"{line.text[:60]}": unknown source(s) {", ".join(unknown)}')
        problems += _numbers_ok(line, entries, max_years)
        problems += _impact_ok(line, entries, profile.summary)
    for text in [resume.headline, *(line.text for line in lines)]:
        if _PLACEHOLDER.search(text):
            problems.append(f'"{text[:60]}": contains a placeholder')
    invented = [s for s in resume.skills if s.lower() not in skill_names]
    if invented:
        problems.append(f"skills not in the profile: {', '.join(invented)}")
    return problems


def check_resume(resume: TailoredResume, profile: Profile, job_id: str) -> TailoredResume:
    """Raise ResumeRejected with every problem, or return the draft with required entries added."""
    problems = validate_resume(resume, profile, job_id)
    if problems:
        raise ResumeRejected("; ".join(problems))
    return add_required_entries(resume, profile)


def add_required_entries(resume: TailoredResume, profile: Profile) -> TailoredResume:
    present = {e.entry_id for e in resume.experience}
    added, notes = [], list(resume.notes)
    for exp in profile.experience:
        recent = exp.end is None or int(exp.end[:4]) >= REQUIRED_SINCE
        if recent and exp.id not in present:
            added.append(ResumeEntry(entry_id=exp.id, highlights=[ResumeLine(text=h, sources=[exp.id])
                                                                    for h in exp.highlights]))
            notes.append(f"Added {exp.title} at {exp.company}: roles since {REQUIRED_SINCE} are always kept.")
    if not added:
        return resume
    return resume.model_copy(update={"experience": [*resume.experience, *added], "notes": notes})


def profile_placeholders(resume: TailoredResume, profile: Profile) -> list[str]:
    """Placeholders in the profile facts this resume would print (name, contact, chosen roles
    and projects, all education). A retry can't fix these; the profile must be edited."""
    found = []
    entries = _entries(profile)

    def scan(where: str, *values) -> None:
        for v in values:
            if isinstance(v, str) and _PLACEHOLDER.search(v):
                found.append(f"{where}: {v!r}")

    scan("name", profile.name)
    if profile.contact:
        scan("contact", profile.contact.email, profile.contact.phone, profile.contact.location,
             *profile.contact.links.values())
    for item in [*resume.experience, *resume.projects]:
        e = entries.get(item.entry_id)
        if e is not None:
            scan(item.entry_id, *[v for v in e.model_dump().values() if isinstance(v, str)])
    for ed in profile.education:
        scan(ed.id, ed.title, ed.institution)
    return found
