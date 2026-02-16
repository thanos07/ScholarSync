# ----- scholarsync/ui.py -----
"""
Streamlit UI helpers for ScholarSync.

✅ Fix included:
- Removed accidental self-import (circular import) that caused:
  ImportError: cannot import name 'apply_theme' from partially initialized module 'scholarsync.ui'

Includes:
- apply_theme(theme): runtime Light/Dark theme via CSS variables
- Material-ish styling layer (rounded cards, subtle borders)
- helper renderers: chips row, evidence cards
"""

from __future__ import annotations

from typing import Any, Dict, List

import streamlit as st


# -------------------------------------------------
# Theme CSS
# -------------------------------------------------
def apply_theme(theme: str) -> None:
    """
    Inject CSS variables + minimal Material-ish styling.
    Theme is runtime-controlled (Light/Dark).

    Variables:
      --bg, --panel, --text, --muted, --border, --primary, --card, --shadow
    """
    theme = (theme or "Light").strip().lower()

    if theme == "dark":
        vars_css = """
        :root {
          --bg: #0b1220;
          --panel: #0f172a;
          --card: #111c33;
          --text: #e5e7eb;
          --muted: #9ca3af;
          --border: rgba(255,255,255,0.10);
          --primary: #60a5fa;
          --shadow: rgba(0,0,0,0.35);
        }
        """
    else:
        vars_css = """
        :root {
          --bg: #f7f7fb;
          --panel: #ffffff;
          --card: #ffffff;
          --text: #0f172a;
          --muted: #6b7280;
          --border: rgba(15,23,42,0.12);
          --primary: #4f46e5;
          --shadow: rgba(15,23,42,0.10);
        }
        """

    css = f"""
    <style>
    {vars_css}

    /* App backgrounds */
    .stApp {{
      background: var(--bg) !important;
      color: var(--text) !important;
    }}

    /* Sidebar */
    section[data-testid="stSidebar"] {{
      background: var(--panel) !important;
      border-right: 1px solid var(--border) !important;
    }}

    /* Typography */
    h1, h2, h3, h4, h5, h6, p, div, span {{
      color: var(--text);
    }}

    .muted {{
      color: var(--muted) !important;
      font-size: 0.95rem;
      line-height: 1.4;
    }}

    /* Inputs */
    .stTextInput input, .stNumberInput input, textarea {{
      background: var(--card) !important;
      border: 1px solid var(--border) !important;
      border-radius: 12px !important;
      color: var(--text) !important;
    }}

    /* Buttons */
    .stButton button {{
      border-radius: 12px !important;
      border: 1px solid var(--border) !important;
      box-shadow: 0 4px 18px var(--shadow);
    }}

    /* Expanders */
    details {{
      background: var(--card) !important;
      border: 1px solid var(--border) !important;
      border-radius: 14px !important;
      padding: 6px 10px !important;
    }}

    /* Card utility */
    .ss-card {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 14px 14px;
      margin: 8px 0px;
      box-shadow: 0 6px 18px var(--shadow);
    }}

    /* Chip utility */
    .ss-chip {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      border: 1px solid var(--border);
      background: var(--card);
      border-radius: 999px;
      padding: 6px 10px;
      margin: 0px 8px 8px 0px;
      font-size: 0.9rem;
      box-shadow: 0 6px 18px var(--shadow);
    }}

    .ss-chip .k {{
      color: var(--muted);
      font-weight: 600;
    }}
    .ss-chip .v {{
      color: var(--text);
      font-weight: 700;
    }}

    a {{
      color: var(--primary) !important;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


# -------------------------------------------------
# Chips Row (metric badges)
# -------------------------------------------------
def chips_row(items: List[Dict[str, str]]) -> None:
    """
    Render a row of metric chips.
    Each item: {"label": "...", "value": "..."}.
    """
    html = []
    for it in items:
        label = str(it.get("label", ""))
        value = str(it.get("value", ""))
        html.append(
            f"<span class='ss-chip'><span class='k'>{label}</span><span class='v'>{value}</span></span>"
        )
    st.markdown("".join(html), unsafe_allow_html=True)


# -------------------------------------------------
# Evidence Cards
# -------------------------------------------------
def evidence_cards(evidence: List[Dict[str, Any]]) -> None:
    """
    Render evidence list as expandable cards:
      - file/page
      - snippet
    """
    if not evidence:
        st.info("No evidence chunks selected.")
        return

    for ev in evidence:
        src = ev.get("source", "unknown")
        page = ev.get("page", "?")
        snippet = ev.get("snippet", "")

        title = f"📄 {src} • page {page}"

        with st.expander(title, expanded=False):
            st.markdown(
                f"""
<div class="ss-card">
  <div class="muted">Evidence snippet</div>
  <div style="margin-top:8px;">{snippet}</div>
</div>
                """.strip(),
                unsafe_allow_html=True,
            )
