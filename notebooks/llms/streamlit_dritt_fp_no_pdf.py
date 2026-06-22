"""Streamlit app to inspect LLM Drittauskunft false-positive predictions
for rows that have no PDF (no object_key). Renders the pre-colored HTML
`clean_text_colored` column instead of a PDF.

Run with:
    streamlit run notebooks/llms/streamlit_dritt_fp_no_pdf.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

CSV_PATH = Path(
    "/Users/melih.gorgulu/Desktop/Projects/aftercourt_automation/notebooks/llms/streamlit_use_dritt_fp_no_object_key.csv"
)

NOT_FP_PATH = Path(
    "/Users/melih.gorgulu/Desktop/Projects/aftercourt_automation/notebooks/llms/not_fp_attachment_ids_no_pdf.txt"
)


@st.cache_data(show_spinner=False)
def load_data(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def load_not_fp_ids() -> set[str]:
    if not NOT_FP_PATH.exists():
        return set()
    with open(NOT_FP_PATH, "r") as f:
        return {line.strip() for line in f if line.strip()}


def append_not_fp_id(attachment_id: str) -> None:
    ids = load_not_fp_ids()
    if attachment_id in ids:
        return
    NOT_FP_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(NOT_FP_PATH, "a") as f:
        f.write(attachment_id + "\n")


def remove_not_fp_id(attachment_id: str) -> None:
    ids = load_not_fp_ids()
    if attachment_id not in ids:
        return
    ids.discard(attachment_id)
    with open(NOT_FP_PATH, "w") as f:
        for x in sorted(ids):
            f.write(x + "\n")


def extract_html(value) -> str:
    """The CSV stores an IPython HTML object's repr or raw HTML string.

    Try to extract the underlying HTML markup. If the value looks like
    `<IPython.core.display.HTML object>` or similar, fall back to empty.
    """
    if not isinstance(value, str):
        return ""
    s = value.strip()
    # IPython HTML objects when converted via str() yield their .data attr
    # which IS the HTML markup, so usually s already contains HTML.
    if s.startswith("<IPython") or s.endswith("object>"):
        return ""
    return s


def main() -> None:
    st.set_page_config(page_title="Dritt FP (no PDF) Inspector", layout="wide")
    st.title("Drittauskunft LLM FP Inspector — No PDF")

    st.markdown(
        '<div style="padding:8px 12px;border-radius:6px;background:#f5f5f5;'
        'border:1px solid #ddd;margin-bottom:12px;font-size:14px;">'
        '<b>Legend:</b> '
        '<span style="color:#2e7d32;font-weight:600;">■ green → drittauskunft</span> &nbsp;|&nbsp; '
        '<span style="color:#c62828;font-weight:600;">■ red → vermögensverzeichnis</span> &nbsp;|&nbsp; '
        '<span style="color:#1565c0;font-weight:600;">■ blue → protokoll</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    if not CSV_PATH.exists():
        st.error(f"CSV not found: {CSV_PATH}")
        return

    df = load_data(CSV_PATH)

    required = ["attachment_id", "document_type", "pred_llm_drittauskunft",
                "q3_32b_is_dritt", "clean_text_colored"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"Missing expected columns: {missing}")
        st.write("Available columns:", df.columns.tolist())
        return

    st.sidebar.header("Navigation")
    st.sidebar.write(f"Total rows: {len(df)}")

    doc_types = ["<all>"] + sorted(df["document_type"].dropna().unique().tolist())
    sel_dt = st.sidebar.selectbox("Filter by document_type", doc_types)
    filtered = df if sel_dt == "<all>" else df[df["document_type"] == sel_dt]
    filtered = filtered.reset_index(drop=True)
    st.sidebar.write(f"Filtered rows: {len(filtered)}")

    if len(filtered) == 0:
        st.info("No rows match the filter.")
        return

    idx = st.sidebar.number_input(
        "Row index",
        min_value=0,
        max_value=len(filtered) - 1,
        value=0,
        step=1,
    )

    col_prev, col_next = st.sidebar.columns(2)
    if col_prev.button("Prev") and idx > 0:
        idx -= 1
    if col_next.button("Next") and idx < len(filtered) - 1:
        idx += 1

    row = filtered.iloc[int(idx)]

    left, right = st.columns([1, 1])

    with left:
        st.subheader("Metadata")
        st.markdown(f"**attachment_id:** `{row['attachment_id']}`")
        st.markdown(f"**document_type:** {row['document_type']}")
        st.markdown(f"**q3_32b_is_dritt:** {row['q3_32b_is_dritt']}")

        saved = load_not_fp_ids()
        att_id = str(row["attachment_id"])
        already = att_id in saved

        c1, c2 = st.columns([1, 1])
        if c1.button(
            "✅ Mark as NOT FP" if not already else "Already marked NOT FP",
            disabled=already,
            type="primary",
            use_container_width=True,
        ):
            append_not_fp_id(att_id)
            st.success(f"Saved {att_id} to {NOT_FP_PATH.name}")
            st.rerun()
        if already and c2.button("Undo", use_container_width=True):
            remove_not_fp_id(att_id)
            st.rerun()

        st.caption(f"Saved NOT-FP ids: {len(saved)} → `{NOT_FP_PATH}`")

        st.subheader("LLM reasoning (pred_llm_drittauskunft)")
        st.text_area(
            label="reasoning",
            value=str(row["pred_llm_drittauskunft"]),
            height=600,
            label_visibility="collapsed",
        )

        with st.expander("clean_text (raw)", expanded=False):
            st.text_area(
                label="clean_text",
                value=str(row.get("clean_text", "")),
                height=400,
                label_visibility="collapsed",
            )

    with right:
        st.subheader("Highlighted text (clean_text_colored)")
        html = extract_html(row["clean_text_colored"])
        if html:
            # Wrap in a scrollable container so long texts don't overflow
            st.markdown(
                f'<div style="max-height:1000px;overflow-y:auto;'
                f'padding:12px;border:1px solid #ddd;border-radius:6px;'
                f'background:#fafafa;">{html}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.warning("No usable HTML in clean_text_colored for this row.")


if __name__ == "__main__":
    main()
