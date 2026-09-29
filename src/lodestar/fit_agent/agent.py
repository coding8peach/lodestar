"""The fit agent: reads a job and the profile through the Lodestar MCP server and judges
each requirement. It returns a FitAnalysis as JSON text and never scores; Python does that.

The agent takes one model. Falling back to another model happens per run, not per
call (see runner.py): a conversation started by one provider can't be continued by another.
"""

import json
from typing import Any

from google.adk.agents import LlmAgent
from google.adk.models.base_llm import BaseLlm
from google.adk.tools.mcp_tool import McpToolset, StdioConnectionParams

from lodestar.mcp_server.params import server_errlog, server_params

FIT_TOOLS = ["get_job", "get_profile"]

# Bump when the instruction changes; eval runs record it so results can be compared by version.
PROMPT_VERSION = "v2c"  # v2b + the hard_constraint field

EXAMPLE_OUTPUT = {
    "job_id": "<the job_id you were given>",
    "requirements": [
        {
            "requirement": "5+ years building backend services in Python",
            "priority": "must_have",
            "evidence": ["exp-example", "proj-example"],
            "match_level": "related",
            "explanation": "Long backend career in Java; Python only in recent projects, so the years are there but not in Python.",
            "hard_constraint": False,
        },
        {
            "requirement": "Experience with Kubernetes",
            "priority": "nice_to_have",
            "evidence": [],
            "match_level": "gap",
            "explanation": "Nothing in the profile mentions container orchestration.",
            "hard_constraint": False,
        },
        {
            "requirement": "Based in the United States",
            "priority": "must_have",
            "evidence": [],
            "match_level": "direct",
            "explanation": "The profile's target locations are in the US.",
            "hard_constraint": True,
        },
    ],
    "summary": "Two or three sentences: main strengths, main gaps, and any hard-constraint problem.",
}

INSTRUCTION = f"""
You are Lodestar's job-fit analyst. For one job posting and one candidate, you judge the fit
requirement by requirement. You never produce a score or a recommendation; code computes those
from your judgments.

1. Call get_job with the job_id you are given, and get_profile. Call both in your first turn.

2. Read the full job description and list its requirements: the skills, experience and
   qualifications the employer asks for.
   - One entry per distinct ask; merge near-duplicates. Keep alternatives the posting offers
     as either/or together ("Perl, Go, or Node.js" is one entry, judged against the closest
     match).
   - Keep each requirement short and close to the posting's wording.
   - Leave out benefits, perks, company background and application instructions.
   - Leave out hiring conditions that are not qualifications: background checks, drug tests,
     reference checks, equal-opportunity statements, "willing to" statements.
   - Keep hard constraints: location or residency, work authorization, time zone, required
     degree or clearance.

3. priority:
   - must_have: presented as required ("required", "must", "you have", "X+ years", or a core
     responsibility the role can't be done without).
   - nice_to_have: framed as a bonus ("preferred", "a plus", "nice to have", "bonus",
     "familiarity with", "exposure to").
   Postings are wish lists: when the wording is unclear, choose nice_to_have unless the role
   clearly can't be done without it.

4. match_level. Base every judgment on what the profile actually states: skills (with depth and
   last_used), experience highlights and technologies, project descriptions and technologies,
   and education. A job title or seniority level alone is not evidence of specific activities
   such as mentoring, leading projects, architecture, observability or testing.
   - direct: the profile states this same skill or experience at the level asked.
   - related: the profile states adjacent experience that transfers. Apply this consistently:
       * backend languages are related to each other (Java, C#, C++, Python, Go, Node.js,
         Ruby, Perl, Kotlin);
       * frontend frameworks are related to each other (React, ExtJS, Flex, Angular, Vue),
         and JavaScript is related to TypeScript;
       * relational databases are related to each other (Oracle, MySQL, SQL Server,
         PostgreSQL, SQLite);
       * cloud and hosting platforms are related to each other (Render, Supabase, Heroku,
         AWS, GCP, Azure);
       * the same skill at a lower depth than asked (course or project level where
         production is asked), or last used long ago (see last_used);
       * an activity the profile doesn't state but a role strongly suggests: at most related,
         and the explanation must say it is inferred from the role.
   - gap: nothing in the profile comes close.
   Skill depth ranks production > project > course. A course-level skill is at most related to
   a requirement for production experience. For "X+ years" requirements, count relevant years
   across the whole career, not just the latest job.

5. Hard constraints are requirements the candidate either meets or doesn't, whatever their
   skills: location or residency, work authorization, time zone, a required degree or security
   clearance. Set "hard_constraint": true on them and false on everything else. Judge location,
   residency, work authorization and time zone against the profile's targets (locations,
   work_modes): direct when the targets satisfy them, gap when they don't, with an empty evidence
   list (the targets are not entries). Judge a required degree or clearance against education.

6. evidence: ids of the profile entries (experience, project or education ids, which look like
   "exp-...", "proj-...", "edu-...", "course-...") whose content actually shows the skill for
   this requirement. A skill's own evidence list is a good source. Cite an entry only if it
   supports this specific requirement: a degree does not show location, and a course does not
   show production use. Never invent an id. Use an empty list for a gap.

7. explanation: one or two sentences on what in the profile supports this level, or what is missing.

8. summary: two or three sentences on the overall picture.

Reply with only a JSON object in exactly this shape, with no markdown fences and no other text:
{json.dumps(EXAMPLE_OUTPUT, indent=2)}
""".strip()


# Fields of get_job the agent never needs: bookkeeping, not posting content.
_JOB_FIELDS_DROPPED = {"url", "source", "classified_by", "fetched_at", "status"}


def _drop_empty(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _drop_empty(v) for k, v in value.items() if v not in (None, "", [], {})}
    if isinstance(value, list):
        return [_drop_empty(v) for v in value]
    return value


def compact_tool_result(tool, args: dict, tool_context, tool_response: dict) -> dict | None:
    """Shrink MCP results before the model sees them.

    An MCP result carries the same data twice: as pretty-printed JSON text (`content`)
    and as `structuredContent`. The model needs it once. Keep the structured copy,
    drop empty fields, and for get_job drop bookkeeping fields.
    """
    if not isinstance(tool_response, dict) or tool_response.get("isError"):
        return None  # errors pass through unchanged so the model sees the message
    data = tool_response.get("structuredContent")
    if not isinstance(data, dict):
        return None
    if getattr(tool, "name", "") == "get_job":
        data = {k: v for k, v in data.items() if k not in _JOB_FIELDS_DROPPED}
    return _drop_empty(data)


def lodestar_mcp_toolset(tools: list[str] | None = None) -> McpToolset:
    """The Lodestar MCP server as a stdio subprocess, limited to the given read-only tools."""
    return McpToolset(
        connection_params=StdioConnectionParams(server_params=server_params(), timeout=60),
        tool_filter=tools or FIT_TOOLS,
        errlog=server_errlog(),  # server logs go to data/logs/mcp_server.log, not our terminal
    )


def _instruction(_ctx) -> str:
    # A callable instruction skips ADK's {state_var} templating, which would otherwise
    # scan the JSON example's braces.
    return INSTRUCTION


def usage_callbacks(usage) -> dict:
    """ADK model callbacks for a UsageRecorder (or none)."""
    if usage is None:
        return {}
    return {"before_model_callback": usage.before_model, "after_model_callback": usage.after_model,
            "on_model_error_callback": usage.on_model_error}


def build_fit_agent(model: BaseLlm, usage=None) -> LlmAgent:
    """`usage`, if given, is a UsageRecorder that sees every model call."""
    callbacks = usage_callbacks(usage)
    return LlmAgent(
        name="fit_agent",
        description="Judges how well the candidate fits one job posting, requirement by requirement.",
        model=model,
        instruction=_instruction,
        tools=[lodestar_mcp_toolset()],
        after_tool_callback=compact_tool_result,
        **callbacks,
    )
