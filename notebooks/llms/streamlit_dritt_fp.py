"""Streamlit app to inspect LLM Drittauskunft false-positive predictions.

Run with:
    streamlit run notebooks/llms/streamlit_dritt_fp.py
"""
from __future__ import annotations

from pathlib import Path

import fitz  # PyMuPDF
import pandas as pd
import streamlit as st

CSV_PATH = Path(
    "/Users/melih.gorgulu/Desktop/Projects/aftercourt_automation/notebooks/llms/streamlit_use_dritt_fp.csv"
)

NOT_FP_PATH = Path(
    "/Users/melih.gorgulu/Desktop/Projects/aftercourt_automation/notebooks/llms/not_fp_attachment_ids.txt"
)

DISPLAY_COLS = [
    "attachment_id",
    "document_type",
    "pred_llm_drittauskunft",
    "q3_32b_is_dritt",
]


@st.cache_data(show_spinner=False)
def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


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


def show_pdf(pdf_path: Path, zoom: float = 2.0) -> None:
    if not pdf_path or not Path(pdf_path).exists():
        st.warning(f"PDF not found: {pdf_path}")
        return
    pdf_path = Path(pdf_path)
    try:
        images = render_pdf_pages(str(pdf_path), zoom=zoom)
    except Exception as e:  # noqa: BLE001
        st.error(f"Failed to render PDF: {e}")
        return

    for i, img_bytes in enumerate(images, start=1):
        st.image(img_bytes, caption=f"Page {i}", use_column_width=True)

    with open(pdf_path, "rb") as f:
        data = f.read()
    st.download_button(
        "Download PDF",
        data=data,
        file_name=pdf_path.name,
        mime="application/pdf",
    )


@st.cache_data(show_spinner=False)
def render_pdf_pages(pdf_path: str, zoom: float = 2.0) -> list[bytes]:
    doc = fitz.open(pdf_path)
    matrix = fitz.Matrix(zoom, zoom)
    pages: list[bytes] = []
    try:
        for page in doc:
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            pages.append(pix.tobytes("png"))
    finally:
        doc.close()
    return pages


def main() -> None:
    st.set_page_config(page_title="Dritt FP Inspector", layout="wide")
    st.title("Drittauskunft LLM False-Positive Inspector")

    if not CSV_PATH.exists():
        st.error(f"CSV not found: {CSV_PATH}")
        return

    df = load_data(CSV_PATH)

    missing = [c for c in DISPLAY_COLS + ["local_dir"] if c not in df.columns]
    if missing:
        st.error(f"Missing expected columns: {missing}")
        st.write("Available columns:", df.columns.tolist())
        return

    st.sidebar.header("Navigation")
    st.sidebar.write(f"Total rows: {len(df)}")

    # Optional filters
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

    zoom = st.sidebar.slider("PDF zoom", min_value=1.0, max_value=4.0, value=2.0, step=0.5)

    row = filtered.iloc[int(idx)]

    left, right = st.columns([1, 1])

    with left:
        st.subheader("Metadata")
        st.markdown(f"**attachment_id:** `{row['attachment_id']}`")
        st.markdown(f"**document_type:** {row['document_type']}")
        st.markdown(f"**q3_32b_is_dritt:** {row['q3_32b_is_dritt']}")
        st.markdown(f"**local_dir:** `{row['local_dir']}`")

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

        with st.expander("clean_text", expanded=False):
            st.text_area(
                label="clean_text",
                value=str(row.get("clean_text", "")),
                height=400,
                label_visibility="collapsed",
            )

    with right:
        st.subheader("PDF")
        pdf_path = row.get("local_dir")
        if isinstance(pdf_path, str) and pdf_path:
            show_pdf(Path(pdf_path), zoom=zoom)
        else:
            st.warning("No local_dir for this row.")


if __name__ == "__main__":
    main()
