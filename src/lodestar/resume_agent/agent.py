"""The resume agent: an ADK LlmAgent that reads the job, the profile and the stored fit
analysis through MCP, and returns a TailoredResume as JSON. It selects, orders and rewords;
Python fills in every fact and checks that nothing is invented (lodestar/resume/validate.py)."""

import json

from google.adk.agents import LlmAgent
from google.adk.models.base_llm import BaseLlm

from lodestar.fit_agent.agent import compact_tool_result, lodestar_mcp_toolset, usage_callbacks
from lodestar.resume.validate import REQUIRED_SINCE

# Bump when the instruction changes; stored with each resume.
PROMPT_VERSION = "r2"  # r2: no puffery; recent project work stays project work
RESUME_TOOLS = ["get_job", "get_profile", "get_fit_analysis"]

EXAMPLE_OUTPUT = {
    "job_id": "<the job_id you were given>",
    "headline": "Backend engineer: Python and Java services on relational databases",
    "summary": {"text": "Backend engineer with 15+ years building data-heavy enterprise applications...",
                "sources": ["exp-example", "proj-example"]},
    "skills": ["Python", "Java", "PostgreSQL"],
    "experience": [
        {"entry_id": "exp-example",
         "highlights": [{"text": "Built the data pipeline for materials experiments in Java on Oracle",
                         "sources": ["exp-example"]}]},
    ],
    "projects": [
        {"entry_id": "proj-example",
         "highlights": [{"text": "Built an MCP server over a question bank, deployed on Render",
                         "sources": ["proj-example"]}]},
    ],
    "notes": ["Kubernetes is required but not in the profile, so it is not claimed."],
}

INSTRUCTION = f"""
You are Lodestar's resume writer. For one job posting, you draft a resume for the candidate
that emphasizes what is relevant to this job, using only what the candidate's profile says.

1. Call get_job, get_profile and get_fit_analysis with the job_id you are given, all in your
   first turn. The fit analysis shows which requirements matter and which profile entries
   support them (evidence ids).

2. You select, order and reword. You never invent. Specifically:
   - Every line you write must cite, in "sources", the ids of the profile entries it is based
     on (experience "exp-...", projects "proj-...", education "edu-..." or "course-...").
   - Do not claim any skill, technology, tool, employer, title, responsibility or result that
     the cited entries don't state. If the job asks for something the profile lacks, leave it
     out and say so in "notes".
   - Do not add numbers (percentages, counts, sizes, team sizes) unless the cited entry states
     that number. "N+ years" is fine if the career dates support it.
   - Do not inflate. Avoid words like "proven track record", "specializing in", "expert",
     "high-scale", "robust", "seasoned" unless the cited entry itself uses them. Describe what
     was done, plainly.
   - Recent project work is project work: put it under projects, and don't present it as years
     of professional experience.
   - Rewording is welcome: lead with the parts that match the job, use the job's terms where
     they honestly describe the same work, keep each line short and concrete.

3. What to include:
   - headline: one line naming the kind of engineer the candidate is, as it fits this job.
   - summary: two or three sentences.
   - skills: the profile's skills that matter for this job, most relevant first. Use the
     skill names exactly as the profile writes them.
   - experience: every role that ended in {REQUIRED_SINCE} or later (or is current); older roles
     only if they help for this job. Two to four highlights for relevant roles, one or two for
     others. Use entry_id only: titles, companies and dates are filled in from the profile.
   - projects: the projects relevant to this job, one to three highlights each.
   - notes: for the candidate, not the resume: requirements you did not claim and why, and
     anything they might add to their profile if it's true.

Reply with only a JSON object in exactly this shape, with no markdown fences and no other text:
{json.dumps(EXAMPLE_OUTPUT, indent=2)}
""".strip()


def _instruction(_ctx) -> str:
    return INSTRUCTION  # callable: skips ADK's {state_var} templating of the JSON braces


def build_resume_agent(model: BaseLlm, usage=None) -> LlmAgent:
    return LlmAgent(
        name="resume_agent",
        description="Drafts a resume tailored to one job, from the candidate's profile only.",
        model=model,
        instruction=_instruction,
        tools=[lodestar_mcp_toolset(RESUME_TOOLS)],
        after_tool_callback=compact_tool_result,
        **usage_callbacks(usage),
    )
