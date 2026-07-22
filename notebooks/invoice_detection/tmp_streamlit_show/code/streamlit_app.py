"""Streamlit app to inspect invoice-detection quality.

For each attachment it shows the model predictions and renders every PDF page.
Pages that the model flagged as belonging to the invoice
(``start_page`` .. ``end_page``, boundaries inclusive) are highlighted with a
green border.

Run with:
    streamlit run streamlit_app.py
"""
import base64
import glob
import os

import fitz  # PyMuPDF
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CODE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CODE_DIR)  # tmp_streamlit_show
CSV_DIR = os.path.join(BASE_DIR, "csv")
PDF_DIR = os.path.join(BASE_DIR, "pdf")

START_COL = "invoice_detection_egvp_start_page"
END_COL = "invoice_detection_egvp_end_page"
INSIDE_COL = "invoice_detection_egvp_is_invoice_inside"

st.set_page_config(page_title="Invoice Detection Quality", layout="wide")

# Use (almost) the full window width so pages can sit side by side.
st.markdown(
    """
    <style>
        .stMainBlockContainer, .block-container {
            max-width: 98% !important;
            padding-left: 2rem;
            padding-right: 2rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner="Loading CSV...")
def load_df(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["attachment_id"] = df["attachment_id"].astype(int)
    return df


def _to_int(value) -> int | None:
    """Best-effort parse of a page number that may be a str/float/'none'."""
    try:
        if pd.isna(value):
            return None
        return int(float(value))
    except (ValueError, TypeError):
        return None


@st.cache_data(show_spinner="Rendering PDF...")
def render_pdf_pages(pdf_path: str, zoom: float = 1.0) -> list[bytes]:
    """Render every page of a PDF into PNG bytes."""
    pages: list[bytes] = []
    with fitz.open(pdf_path) as doc:
        matrix = fitz.Matrix(zoom, zoom)
        for page in doc:
            pix = page.get_pixmap(matrix=matrix)
            pages.append(pix.tobytes("png"))
    return pages


def _page_html(png_bytes: bytes, page_number: int, is_invoice: bool, basis: str) -> str:
    """Return the HTML for a single page card (image + label)."""
    b64 = base64.b64encode(png_bytes).decode("utf-8")
    if is_invoice:
        border = "6px solid #2ecc40"
        label = f"Page {page_number} — INVOICE"
        label_color = "#2ecc40"
    else:
        border = "1px solid #cccccc"
        label = f"Page {page_number}"
        label_color = "#888888"
    return f"""
        <div style="flex:0 0 {basis}; text-align:center; margin:8px;">
            <div style="color:{label_color}; font-weight:600; margin-bottom:4px;">{label}</div>
            <img src="data:image/png;base64,{b64}"
                 style="border:{border}; border-radius:4px; width:100%; box-shadow:0 1px 4px rgba(0,0,0,0.15);" />
        </div>
    """


def show_pages_grid(pages: list[bytes], start_page, end_page, per_row: int) -> None:
    """Render all pages side by side, ``per_row`` pages per row, fitting to width."""
    # width as a percentage so the row always fills the available width
    basis = f"calc({100 / per_row:.4f}% - 16px)"
    cards = []
    for i, png in enumerate(pages, start=1):
        is_invoice = (
            start_page is not None
            and end_page is not None
            and start_page <= i <= end_page
        )
        cards.append(_page_html(png, i, is_invoice, basis))
    st.markdown(
        f"""
        <div style="display:flex; flex-wrap:wrap; justify-content:flex-start; align-items:flex-start;">
            {''.join(cards)}
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.title("📄 Invoice Detection Quality Analysis")

    # ---- Sidebar: pick a dataset --------------------------------------
    st.sidebar.header("Dataset")
    csv_files = sorted(glob.glob(os.path.join(CSV_DIR, "*.csv")))
    if not csv_files:
        st.error(f"No CSV files found in: {CSV_DIR}")
        return
    csv_names = [os.path.basename(p) for p in csv_files]
    selected_csv = st.sidebar.selectbox("Dataset CSV", csv_names)
    csv_path = os.path.join(CSV_DIR, selected_csv)

    df = load_df(csv_path)

    # ---- Sidebar: pick an attachment ----------------------------------
    st.sidebar.header("Attachments")
    only_inside = st.sidebar.checkbox("Only is_invoice_inside == True", value=False)
    view_df = df[df[INSIDE_COL].astype(str) == "True"] if only_inside else df

    pages_per_row = st.sidebar.slider("Pages per row", 1, 5, 2)

    show_text = st.sidebar.toggle("Show OCR text", value=False)

    if st.sidebar.button("🔄 Reload CSV"):
        load_df.clear()
        st.rerun()

    ids = view_df["attachment_id"].tolist()
    if not ids:
        st.warning("No attachments match the current filter.")
        return

    # Keep the current index in session state so the nav buttons work.
    # Reset when the dataset changes.
    if st.session_state.get("dataset") != selected_csv:
        st.session_state.dataset = selected_csv
        st.session_state.idx = 0
    if "idx" not in st.session_state or st.session_state.idx >= len(ids):
        st.session_state.idx = 0

    # ---- Sidebar navigation buttons -----------------------------------
    nav_next, nav_prev = st.sidebar.columns(2)
    if nav_next.button("Next ➡️", use_container_width=True):
        st.session_state.idx = (st.session_state.idx + 1) % len(ids)
    if nav_prev.button("⬅️ Previous", use_container_width=True):
        st.session_state.idx = (st.session_state.idx - 1) % len(ids)
    current_id = ids[st.session_state.idx]
    st.markdown(
        f"### Attachment `{current_id}`\n\n"
        f"**Document {st.session_state.idx + 1} / {len(ids)}**"
    )

    attachment_id = st.sidebar.selectbox(
        "Select attachment_id",
        ids,
        index=st.session_state.idx,
        format_func=lambda a: str(a),
    )
    # Sync session index if the user picked from the selectbox instead.
    st.session_state.idx = ids.index(attachment_id)

    row = df[df["attachment_id"] == attachment_id].iloc[0]
    start_page = _to_int(row.get(START_COL))
    end_page = _to_int(row.get(END_COL))
    is_inside = str(row.get(INSIDE_COL))

    # ---- Predictions summary ------------------------------------------
    st.subheader(f"Attachment {attachment_id}")
    c1, c2, c3 = st.columns(3)
    c1.metric("Is Invoice Inside", is_inside)
    c2.metric("Start Page", start_page if start_page is not None else "—")
    c3.metric("End Page", end_page if end_page is not None else "—")

    with st.expander("All predictions for this attachment"):
        st.dataframe(row.to_frame(name="value"), use_container_width=True)

    # ---- Optional OCR text --------------------------------------------
    if show_text:
        text = row.get("text")
        if pd.isna(text) or text is None or str(text).strip() == "":
            st.info("No text available for this attachment.")
        else:
            st.text_area("OCR text", str(text), height=300)

    # ---- PDF rendering ------------------------------------------------
    pdf_path = os.path.join(PDF_DIR, f"{attachment_id}.pdf")
    if not os.path.exists(pdf_path):
        st.error(f"PDF not found: {pdf_path}\nRun download_pdfs.py first.")
        return

    pages = render_pdf_pages(pdf_path)
    st.caption(f"{len(pages)} page(s) — invoice pages are highlighted with a green border.")

    show_pages_grid(pages, start_page, end_page, pages_per_row)


if __name__ == "__main__":
    main()
