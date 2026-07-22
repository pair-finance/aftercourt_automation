"""Streamlit app to review invoice-page detection on *unknown* documents.

For each sampled attachment it shows the document PDF next to all of its
predictions (highlighting the ``invoice_detection_egvp`` output) and lets the
reviewer record a verdict on whether the invoice-page detection is correct.

Prerequisites (produced by ``check_invoice_pages_for_unknown_docs.ipynb``):
    - PDFs downloaded to ``assets/pdfs/tmp/unknown_rejected/<attachment_id>.pdf``
    - combined predictions CSV ``unknown_rejected_all_predictions.csv`` in that dir

Run with:
    streamlit run notebooks/classification/invoice_page_detection/review_invoice_pages_unknown_docs_streamlit.py
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import fitz  # PyMuPDF
import pandas as pd
import streamlit as st

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
BASE_DIR = Path(
    "/Users/melih.gorgulu/Desktop/Projects/aftercourt_automation/notebooks/classification/invoice_page_detection"
)
PDF_DIR = Path(
    "/Users/melih.gorgulu/Desktop/Projects/aftercourt_automation/assets/pdfs/tmp/unknown_rejected"
)
SAMPLE_CSV = BASE_DIR / "invoice_page_predictions_unknown_docs_sample.csv"
PREDICTIONS_CSV = PDF_DIR / "unknown_rejected_all_predictions.csv"
REVIEW_CSV = BASE_DIR / "invoice_page_review_results.csv"

VERDICT_OPTIONS = ["unreviewed", "correct", "incorrect", "unsure"]


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def load_predictions(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # strip the single-quote wrappers stored around values (e.g. "'True'" -> "True")
    df["value_clean"] = df["value"].astype(str).str.strip("'")
    return df


@st.cache_data(show_spinner=False)
def load_sample(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def load_reviews() -> dict[str, dict]:
    if not REVIEW_CSV.exists():
        return {}
    df = pd.read_csv(REVIEW_CSV)
    return {str(r["attachment_id"]): r.to_dict() for _, r in df.iterrows()}


def save_review(attachment_id: str, source: str, verdict: str, notes: str) -> None:
    reviews = load_reviews()
    reviews[str(attachment_id)] = {
        "attachment_id": attachment_id,
        "source": source,
        "verdict": verdict,
        "notes": notes,
        "reviewed_at": datetime.utcnow().isoformat(timespec="seconds"),
    }
    out = pd.DataFrame(reviews.values())
    REVIEW_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(REVIEW_CSV, index=False)


@st.cache_data(show_spinner=False)
def render_pdf_pages(
    pdf_path: str, zoom: float = 2.0, invoice_pages: tuple[int, ...] = ()
) -> list[bytes]:
    """Render PDF pages to PNG bytes, drawing a green border on invoice pages.

    ``invoice_pages`` holds 1-based page numbers (start..end inclusive) to mark.
    """
    invoice_set = set(invoice_pages)
    doc = fitz.open(pdf_path)
    matrix = fitz.Matrix(zoom, zoom)
    pages: list[bytes] = []
    try:
        for page in doc:
            if (page.number + 1) in invoice_set:
                inset = 4.0
                border = fitz.Rect(
                    page.rect.x0 + inset,
                    page.rect.y0 + inset,
                    page.rect.x1 - inset,
                    page.rect.y1 - inset,
                )
                page.draw_rect(border, color=(0, 0.7, 0), width=8)
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            pages.append(pix.tobytes("png"))
    finally:
        doc.close()
    return pages


def show_pdf(
    pdf_path: Path,
    zoom: float,
    invoice_pages: tuple[int, ...],
    view_mode: str,
    page_width: int,
    fit_width: bool = True,
    cols_per_row: int = 2,
) -> None:
    if not pdf_path or not Path(pdf_path).exists():
        st.warning(f"PDF not found: {pdf_path}")
        return
    try:
        images = render_pdf_pages(str(pdf_path), zoom=zoom, invoice_pages=invoice_pages)
    except Exception as e:  # noqa: BLE001
        st.error(f"Failed to render PDF: {e}")
        return

    n_pages = len(images)
    invoice_set = set(invoice_pages)

    if view_mode == "All pages":
        page_numbers = list(range(1, n_pages + 1))
    elif view_mode == "Single page":
        default_page = invoice_pages[0] if invoice_pages else 1
        sel = st.number_input(
            "Page", min_value=1, max_value=n_pages, value=min(default_page, n_pages), step=1
        )
        page_numbers = [int(sel)]
    else:  # "Invoice pages"
        page_numbers = list(invoice_pages) if invoice_pages else [1]

    page_numbers = [p for p in page_numbers if 1 <= p <= n_pages]
    width_arg = "stretch" if fit_width else page_width
    cols_per_row = max(1, int(cols_per_row))

    # Lay pages out in a grid, cols_per_row per row.
    for start in range(0, len(page_numbers), cols_per_row):
        row_pages = page_numbers[start : start + cols_per_row]
        cols = st.columns(cols_per_row)
        for col, p in zip(cols, row_pages):
            caption = f"Page {p} \U0001f7e2 invoice" if p in invoice_set else f"Page {p}"
            col.image(images[p - 1], caption=caption, width=width_arg)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _pred_value(att_preds: pd.DataFrame, model_name: str, subtype: str) -> str:
    vals = att_preds[
        (att_preds["model_name"] == model_name) & (att_preds["subtype"] == subtype)
    ]["value_clean"].values
    return vals[0] if len(vals) else "N/A"


def _to_int(value: str) -> int | None:
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return None


def _invoice_page_range(att_preds: pd.DataFrame) -> tuple[int, ...]:
    """1-based page numbers start..end (inclusive) for the invoice, or empty."""
    start = _to_int(_pred_value(att_preds, "invoice_detection_egvp", "start_page"))
    end = _to_int(_pred_value(att_preds, "invoice_detection_egvp", "end_page"))
    if start is None or end is None or start < 1 or end < start:
        return ()
    return tuple(range(start, end + 1))


# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #
def main() -> None:
    st.set_page_config(page_title="Invoice-Page Review (unknown docs)", layout="wide")
    st.title("Invoice-Page Detection Review — Unknown Documents")

    for label, path in [("sample", SAMPLE_CSV), ("predictions", PREDICTIONS_CSV)]:
        if not path.exists():
            st.error(f"Missing {label} file: {path}")
            st.info("Run the notebook cells first to generate the PDFs and predictions CSV.")
            return

    sample = load_sample(str(SAMPLE_CSV))
    preds = load_predictions(str(PREDICTIONS_CSV))
    reviews = load_reviews()

    # Attachments to review, keyed off the sample (keeps the intended source split).
    att_source = (
        sample[["attachment_id", "source"]]
        .drop_duplicates("attachment_id")
        .reset_index(drop=True)
    )

    # -------------------- Sidebar navigation / filters -------------------- #
    st.sidebar.header("Navigation")

    sources = ["<all>"] + sorted(att_source["source"].dropna().unique().tolist())
    sel_source = st.sidebar.selectbox("Source", sources, key="filter_source")

    verdict_filter = st.sidebar.selectbox(
        "Verdict filter", ["<all>", "unreviewed", "correct", "incorrect", "unsure"],
        key="filter_verdict",
    )

    filtered = att_source if sel_source == "<all>" else att_source[att_source["source"] == sel_source]

    def _verdict_of(att_id: str) -> str:
        return str(reviews.get(str(att_id), {}).get("verdict", "unreviewed"))

    if verdict_filter != "<all>":
        filtered = filtered[filtered["attachment_id"].map(_verdict_of) == verdict_filter]

    filtered = filtered.reset_index(drop=True)

    n_total = len(att_source)
    n_reviewed = sum(1 for a in att_source["attachment_id"] if str(a) in reviews)
    st.sidebar.metric("Reviewed", f"{n_reviewed} / {n_total}")
    st.sidebar.write(f"Matching filter: {len(filtered)}")

    if len(filtered) == 0:
        st.info("No attachments match the current filter.")
        return

    idx = st.sidebar.number_input(
        "Index", min_value=0, max_value=len(filtered) - 1, value=0, step=1, key="nav_index"
    )
    zoom = st.sidebar.slider("PDF zoom", min_value=1.0, max_value=4.0, value=2.0, step=0.5, key="pdf_zoom")
    view_mode = st.sidebar.radio(
        "PDF view", ["All pages", "Invoice pages", "Single page"], index=0, key="pdf_view"
    )
    cols_per_row = st.sidebar.slider(
        "Pages per row", min_value=1, max_value=4, value=2, step=1, key="pdf_cols_per_row"
    )
    fit_width = st.sidebar.checkbox("Fit page to width", value=True, key="pdf_fit_width")
    page_width = st.sidebar.slider(
        "Page width (px)", min_value=300, max_value=1400, value=800, step=50,
        key="pdf_page_width", disabled=fit_width,
    )

    row = filtered.iloc[int(idx)]
    att_id = str(row["attachment_id"])
    source = str(row["source"])
    att_preds = preds[preds["attachment_id"].astype(str) == att_id]

    pdf_path = PDF_DIR / f"{att_id}.pdf"
    if not att_preds.empty and "pdf_path" in att_preds.columns:
        stored = str(att_preds["pdf_path"].iloc[0])
        if stored and stored != "nan":
            pdf_path = Path(stored)

    # --------------------- Sidebar: document + invoice info --------------- #
    st.sidebar.markdown("---")
    st.sidebar.subheader("Document")
    st.sidebar.markdown(f"**attachment_id:** `{att_id}`")
    st.sidebar.markdown(f"**source:** {source}")
    if not att_preds.empty and "file_name" in att_preds.columns:
        st.sidebar.markdown(f"**file_name:** {att_preds['file_name'].iloc[0]}")
    st.sidebar.markdown(f"**current verdict:** `{_verdict_of(att_id)}`")

    st.sidebar.subheader("invoice_detection_egvp")
    sc1, sc2, sc3 = st.sidebar.columns(3)
    sc1.metric("is_invoice_inside", _pred_value(att_preds, "invoice_detection_egvp", "is_invoice_inside"))
    sc2.metric("start_page", _pred_value(att_preds, "invoice_detection_egvp", "start_page"))
    sc3.metric("end_page", _pred_value(att_preds, "invoice_detection_egvp", "end_page"))

    # ----------------------------- Labeling ------------------------------- #
    st.subheader("Labeling")
    invoice_pages = _invoice_page_range(att_preds)
    existing = reviews.get(att_id, {})
    default_verdict = str(existing.get("verdict", "unreviewed"))
    default_notes = str(existing.get("notes", "") or "")

    with st.form(key=f"review_{att_id}"):
        verdict = st.radio(
            "Is the invoice-page detection correct?",
            VERDICT_OPTIONS,
            index=VERDICT_OPTIONS.index(default_verdict) if default_verdict in VERDICT_OPTIONS else 0,
            horizontal=True,
        )
        notes = st.text_area("Notes", value=default_notes, height=80)
        submitted = st.form_submit_button("💾 Save verdict", type="primary")
        if submitted:
            save_review(att_id, source, verdict, notes)
            st.success(f"Saved verdict '{verdict}' for {att_id}")
            st.rerun()

    # -------------------- Optional details (collapsed) -------------------- #
    with st.expander("All predictions", expanded=False):
        if att_preds.empty:
            st.warning("No predictions found for this attachment.")
        else:
            st.dataframe(
                att_preds[["model_name", "type", "subtype", "value_clean"]]
                .rename(columns={"value_clean": "value"})
                .sort_values(["model_name", "subtype"])
                .reset_index(drop=True),
                use_container_width=True,
                height=320,
            )

    if not att_preds.empty and "text" in att_preds.columns:
        with st.expander("Textract text", expanded=False):
            st.text_area(
                "text",
                value=str(att_preds["text"].iloc[0]),
                height=300,
                label_visibility="collapsed",
            )

    # ----------------------------- Document ------------------------------- #
    st.subheader("Document")
    if invoice_pages:
        st.caption(
            f"\U0001f7e2 Invoice pages (green border): {invoice_pages[0]}\u2013{invoice_pages[-1]}"
        )
    show_pdf(
        pdf_path,
        zoom=zoom,
        invoice_pages=invoice_pages,
        view_mode=view_mode,
        page_width=page_width,
        fit_width=fit_width,
        cols_per_row=cols_per_row,
    )


if __name__ == "__main__":
    main()
