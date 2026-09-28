"""The four pages. Each is a plain function, so the app and tests can run them."""

import pandas as pd
import streamlit as st

from lodestar.app import service
from lodestar.db import DecisionError
from lodestar.ui.components import RECOMMENDATION_LABEL, money, requirements_table

LABEL_TO_REC = {v: k for k, v in RECOMMENDATION_LABEL.items()}


# --- Review ---------------------------------------------------------------------------

def _pick_next(ids: list[str], current: str) -> str | None:
    """The job after `current` in the list (or the one before, if it was last)."""
    if current not in ids:
        return ids[0] if ids else None
    i = ids.index(current)
    rest = ids[i + 1:] or ids[:i]
    return rest[0] if rest else None


def _decide(job_id: str, decision: str, note: str, ids: list[str]) -> None:
    try:
        service.record_decision(job_id, decision, note or None)
    except DecisionError as e:
        st.error(str(e))
        return
    st.session_state["review_selected"] = _pick_next(ids, job_id)
    st.session_state["review_table_version"] = st.session_state.get("review_table_version", 0) + 1
    st.toast({"approve": "Approved", "reject": "Rejected", "skip": "Skipped"}[decision])
    st.rerun()


def _detail(job_id: str, ids: list[str]) -> None:
    d = service.job_detail(job_id)
    job, fit = d.job, d.fit
    st.subheader(job.title)
    where = ", ".join(x for x in (job.company, job.location) if x)
    st.caption(where or "")
    st.link_button("Open the posting", job.url)

    cols = st.columns(4)
    cols[0].metric("Recommendation", RECOMMENDATION_LABEL[fit.recommendation])
    cols[1].metric("Skill fit", f"{fit.overall_score:.2f}")
    cols[2].metric("Model", d.model or "–")
    cols[3].metric("Analyzed", d.analyzed_at.strftime("%b %d") if d.analyzed_at else "–")
    if d.blocked_by:
        st.error("Can't apply: " + "; ".join(d.blocked_by))

    requirements_table(fit)
    st.markdown(f"**Summary.** {fit.analysis.summary}")
    if d.similar:
        st.caption("Similar postings: " + "; ".join(f"{s.title} ({s.status})" for s in d.similar))
    with st.expander("Full job description"):
        st.text(job.description)

    note = st.text_input("Note (optional)", key=f"note_{job_id}")
    a, r, s, _ = st.columns([1, 1, 1, 4])
    if a.button("Approve", type="primary", key="approve"):
        _decide(job_id, "approve", note, ids)
    if r.button("Reject", key="reject"):
        _decide(job_id, "reject", note, ids)
    if s.button("Skip for now", key="skip"):
        _decide(job_id, "skip", note, ids)


def review_page() -> None:
    st.title("Review")
    everything = service.review_queue()
    blocked = [i for i in everything if i.recommendation == "blocked"]

    with st.sidebar:
        st.header("Filter")
        chosen = st.multiselect("Recommendation", ["Top pick", "Possible"], default=["Top pick", "Possible"])
        companies = sorted({i.company for i in everything if i.company and i.recommendation != "blocked"})
        company = st.selectbox("Company", ["All companies", *companies])
        search = st.text_input("Search titles")

    items = service.review_queue(recommendations=[LABEL_TO_REC[c] for c in chosen] or ["top_pick", "possible"],
                                 company=None if company == "All companies" else company, search=search)
    items = [i for i in items if i.recommendation != "blocked"]

    if not items:
        st.info("Nothing waiting for a decision. Analyze queued jobs with `uv run lodestar analyze`.")
    else:
        ids = [i.job_id for i in items]
        selected = st.session_state.get("review_selected")
        if selected not in ids:
            selected = ids[0]
        df = pd.DataFrame([{
            "Recommendation": RECOMMENDATION_LABEL[i.recommendation], "Skill fit": round(i.score, 2),
            "Company": i.company or "?", "Title": i.title, "Location": i.location or "",
            "Analyzed": i.analyzed_at.strftime("%b %d"),
        } for i in items])
        st.caption(f"{len(items)} waiting for a decision. Select a row to see the analysis.")
        event = st.dataframe(
            df, hide_index=True, width="stretch", on_select="rerun", selection_mode="single-row",
            selection_default={"selection": {"rows": [ids.index(selected)]}},
            column_config={
                "Recommendation": st.column_config.TextColumn(width="small"),
                "Skill fit": st.column_config.NumberColumn(width="small"),
                "Company": st.column_config.TextColumn(width="small"),
                "Title": st.column_config.TextColumn(width="large"),
                "Location": st.column_config.TextColumn(width="medium"),
                "Analyzed": st.column_config.TextColumn(width="small"),
            },
            key=f"review_table_{st.session_state.get('review_table_version', 0)}",
        )
        rows = event.selection.rows if event and event.selection else []
        if rows:
            selected = ids[rows[0]]
        st.session_state["review_selected"] = selected
        st.divider()
        _detail(selected, ids)

    if blocked:
        st.divider()
        with st.expander(f"Blocked: {len(blocked)} job(s) you can't apply to"):
            for i in blocked:
                st.markdown(f"**{i.company or '?'}**, {i.title}: {'; '.join(i.blocked_by)}")
            sure = st.checkbox("Reject all of these, with the reason as the note", key="confirm_blocked")
            if st.button("Reject blocked jobs", disabled=not sure, key="reject_blocked"):
                result = service.reject_blocked(i.job_id for i in blocked)
                st.toast(f"Rejected {result.done}")
                st.rerun()


# --- Queue ---------------------------------------------------------------------------

def _escape(text: str) -> str:
    """Streamlit markdown treats $...$ as math; escape dollar signs in plain text."""
    return text.replace("$", "\\$")


def _show_last_run() -> None:
    """Result of the last Find / Analyze click, kept across the rerun that refreshes the numbers."""
    last = st.session_state.pop("last_run", None)
    if not last:
        return
    kind, label, lines, ok = last
    with st.status(label, state="complete" if ok else "error", expanded=bool(lines)):
        for line in lines:
            st.write(_escape(line))


def _discover() -> None:
    with st.spinner("Listing watched job boards..."):
        summary = service.discover()
    new = summary.listed - summary.seen
    lines = [f"Queued: {c or '?'}, {t}" for c, t, _ in summary.queued_jobs]
    filtered = new - len(summary.queued_jobs)
    if filtered:
        reasons = ", ".join(f"{r} {n}" for r, n in summary.skipped.items())
        lines.append(f"Filtered out {filtered}: {reasons}")
    lines += [f"{b.company}: error, {b.error}" for b in summary.boards if b.error]
    label = f"Found {new} new posting(s): queued {len(summary.queued_jobs)}, filtered out {filtered}"
    st.session_state["last_run"] = ("discover", label, lines, True)


def _analyze(budget: int) -> None:
    lines: list[str] = []
    with st.status(f"Analyzing {budget} job(s)...", expanded=True):
        def show(item) -> None:
            line = (f"Failed: {item.company or '?'}, {item.title}. {item.error}" if item.error else
                    f"{RECOMMENDATION_LABEL[item.recommendation]} {item.score:.2f}: {item.company or '?'}, {item.title}")
            lines.append(line)
            st.write(_escape(line))
        batch = service.analyze_next(budget, on_progress=show)
    label = f"Analyzed {batch.ok}" + (f", {batch.failed} failed" if batch.failed else "")
    label += (". Stopped early: " + batch.stopped) if batch.stopped else (
        ". Ready on the Review page." if batch.ok else "")
    st.session_state["last_run"] = ("analyze", _escape(label), lines, not batch.stopped)


def _run_bar() -> None:
    """One row: Find new postings | jobs to analyze | Analyze | status."""
    from lodestar.fit_agent.client import agent_status

    running, status = agent_status()
    money_today = service.spending()
    find, count, go, info = st.columns([1.3, 0.9, 1.3, 4.5], vertical_alignment="center")
    if find.button("Find new postings", key="discover", help="Lists your watched boards. No LLM, no cost."):
        _discover()
        st.session_state["queue_table_version"] = st.session_state.get("queue_table_version", 0) + 1
        st.rerun()
    budget = count.number_input("Jobs to analyze", min_value=1, max_value=20, value=3, key="budget",
                                label_visibility="collapsed")
    if go.button(f"Analyze next {budget}", type="primary", key="analyze", disabled=not running):
        _analyze(int(budget))
        st.session_state["queue_table_version"] = st.session_state.get("queue_table_version", 0) + 1
        st.rerun()
    agent = "Fit agent running." if running else status + "."
    info.caption(_escape(f"{agent} Paid models today: ${money_today.paid_today_usd:.2f} spent of "
                         f"${money_today.paid_limit_usd:.2f} allowed."))
    _show_last_run()


def queue_page() -> None:
    st.title("Queue")
    f = service.funnel()
    cols = st.columns(6)
    cols[0].metric("Postings seen", f.discovered)
    cols[1].metric("Filtered out", f.discovery_skipped)
    cols[2].metric("Queued", f.count("queued"))
    cols[3].metric("Waiting for you", f.count("analyzed", "awaiting_review"))
    cols[4].metric("Approved", f.count("approved"))
    cols[5].metric("Rejected", f.count("rejected"))

    _run_bar()

    try:
        ranked = service.ranked_queue()
    except FileNotFoundError as e:
        st.error(f"Can't rank the queue: {e}. The ranking reads your profile (data/profile.yaml).")
        return
    if not ranked:
        st.info("The queue is empty. Use Find new postings to check your watched boards.")
        return
    st.caption(f"{len(ranked)} distinct roles, in the order Analyze takes them. Hover a title to see all of it.")
    df = pd.DataFrame([{
        "Rank": n, "Score": q.score, "Company": q.company or "?", "Title": q.title, "Location": q.location or "",
        "Matched": ", ".join(q.matched), "Similar": q.similar_count or None,  # blank rather than 0
        "Posted": q.date_posted.isoformat() if q.date_posted else "", "Posting": q.url,
    } for n, q in enumerate(ranked, 1)])
    event = st.dataframe(
        df, hide_index=True, width="stretch", on_select="rerun", selection_mode="multi-row",
        column_config={
            "Rank": st.column_config.NumberColumn(width=50),
            "Score": st.column_config.NumberColumn(width=55),
            "Company": st.column_config.TextColumn(width="small"),
            "Title": st.column_config.TextColumn(width="large"),
            "Location": st.column_config.TextColumn(width="medium"),
            "Matched": st.column_config.TextColumn(width="medium"),
            "Similar": st.column_config.NumberColumn(width=60),
            "Posted": st.column_config.TextColumn(width=95),
            "Posting": st.column_config.LinkColumn(display_text="Open", width=60),
        },
        key=f"queue_table_{st.session_state.get('queue_table_version', 0)}")
    rows = event.selection.rows if event and event.selection else []
    if rows:
        reason = st.text_input("Reason", value="not interested", key="dismiss_reason")
        if st.button(f"Dismiss {len(rows)} selected without analyzing", key="dismiss"):
            result = service.dismiss([ranked[r].job_id for r in rows], reason)
            st.session_state["queue_table_version"] = st.session_state.get("queue_table_version", 0) + 1
            st.toast(f"Dismissed {result.done}")
            st.rerun()


# --- Decisions -------------------------------------------------------------------------

def _decision_table(items) -> None:
    if not items:
        st.info("None yet.")
        return
    df = pd.DataFrame([{
        "Decided": i.decided_at.strftime("%b %d"), "Company": i.company or "?", "Title": i.title,
        "Note": ("dismissed: " if i.dismissed else "") + (i.note or ""),
        "Agent": RECOMMENDATION_LABEL.get(i.recommendation, "not analyzed"),
        "Skill fit": round(i.score, 2) if i.score is not None else None, "Posting": i.url,
    } for i in items])
    st.dataframe(df, hide_index=True, width="stretch",
                 column_config={"Posting": st.column_config.LinkColumn(display_text="Open")})


def decisions_page() -> None:
    st.title("Decisions")
    approved, rejected = st.tabs(["Approved: your apply list", "Rejected"])
    with approved:
        _decision_table(service.decisions("approve"))
    with rejected:
        _decision_table(service.decisions("reject"))


# --- Usage --------------------------------------------------------------------------------

def usage_page() -> None:
    st.title("Usage")
    by = st.radio("Group by", ["run", "day", "model"], horizontal=True, key="usage_by")
    rows = service.usage(by)
    if not rows:
        st.info("No model calls recorded yet.")
        return
    cols = st.columns(4)
    cols[0].metric("Calls", sum(r.calls for r in rows))
    cols[1].metric("Tokens", f"{sum(r.input_tokens + r.output_tokens for r in rows):,}")
    priced = [r.list_cost_usd for r in rows if r.list_cost_usd is not None]
    paid = [r.cost_usd for r in rows if r.cost_usd is not None]
    cols[2].metric("List cost", money(sum(priced)) if priced else "–")
    cols[3].metric("You paid", money(sum(paid)) if paid else "–")
    df = pd.DataFrame([{
        by.capitalize(): r.key, "Calls": r.calls, "Failed": r.failed_calls, "Input tokens": r.input_tokens,
        "Output tokens": r.output_tokens, "List cost": money(r.list_cost_usd), "You paid": money(r.cost_usd),
    } for r in rows])
    st.dataframe(df, hide_index=True, width="stretch")
    if by == "day":
        chart = pd.DataFrame({"day": [r.key for r in rows], "list cost ($)": [r.list_cost_usd or 0 for r in rows]})
        st.bar_chart(chart.set_index("day"))
    st.caption("List cost is tokens times list price. You paid is $0 for models marked free_tier "
               "in config/prices.yaml. A dash means the price is unknown.")
