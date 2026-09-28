"""Small display helpers shared by the pages."""

import pandas as pd
import streamlit as st

from lodestar.schemas.fit import FitResult

RECOMMENDATION_LABEL = {"top_pick": "Top pick", "possible": "Possible", "blocked": "Blocked"}
MATCH_COLOR = {"direct": "#d9eedb", "related": "#fbeccc", "gap": "#f5d5d5"}   # green / amber / rose


def money(value: float | None) -> str:
    return "–" if value is None else f"${value:.4f}"


def requirements_table(fit: FitResult) -> None:
    rows = [{
        "Match": m.match_level,
        "Priority": ("must have" if m.priority == "must_have" else "nice to have")
                    + (", hard constraint" if m.hard_constraint else ""),
        "Requirement": m.requirement,
        "Why": m.explanation,
        "Evidence": ", ".join(m.evidence),
    } for m in fit.analysis.requirements]
    df = pd.DataFrame(rows)
    styled = df.style.apply(
        lambda row: [f"background-color: {MATCH_COLOR.get(row['Match'], 'white')}" if col == "Match" else ""
                     for col in df.columns], axis=1)
    st.dataframe(styled, hide_index=True, width="stretch",
                 column_config={"Why": st.column_config.TextColumn(width="large")})
