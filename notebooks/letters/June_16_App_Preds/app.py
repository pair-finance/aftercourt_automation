"""Streamlit app to review predictions alongside their PDF documents.

Run with:
    streamlit run notebooks/letters/June_16_App_Preds/app.py
"""

import os

import fitz  # PyMuPDF
import pandas as pd
import requests
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_CSV = os.path.join(BASE_DIR, "egvp_results_16June_positives.csv")
SAMPLED_CSV = os.path.join(BASE_DIR, "sampled_data.csv")
PDF_DIR = "/Users/melih.gorgulu/Desktop/Projects/aftercourt_automation/assets/pdfs/tmp"

os.makedirs(PDF_DIR, exist_ok=True)


@st.cache_data
def load_data() -> pd.DataFrame:
    """Load the sampled data, regenerating it from source if the export is missing."""
    if os.path.exists(SAMPLED_CSV):
        return pd.read_csv(SAMPLED_CSV)

    data = pd.read_csv(SOURCE_CSV)
    data = data[data["predicted_types"] != "['pfub_erlass(llm)', 'drittauskunft']"]
    data = data[data["predicted_types"] != "['pfub_erlass(llm)']"]
    n = 40
    sampled = (
        data.groupby("predicted_types")
        .apply(lambda x: x.sample(min(n, len(x)), replace=True))
        .reset_index(drop=True)
    )
    return sampled


def pdf_path_for(row: pd.Series, fallback_idx: int) -> str:
    """Return the local file path where a row's PDF should live."""
    attachment_id = row.get("attachment_id")
    if pd.isna(attachment_id) or attachment_id in (None, ""):
        attachment_id = f"row_{fallback_idx}"
    return os.path.join(PDF_DIR, f"{attachment_id}.pdf")


def download_pdf(url: str, dest_path: str) -> str:
    """Download the PDF at `url` to `dest_path` (cached on disk). Returns the path."""
    if not os.path.exists(dest_path):
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        with open(dest_path, "wb") as f:
            f.write(resp.content)
    return dest_path


@st.cache_resource(show_spinner=False)
def predownload_all(df: pd.DataFrame) -> dict:
    """Download every PDF up front. Returns {row_index: local_path or None}."""
    paths: dict = {}
    rows = list(df.iterrows())
    progress = st.progress(0.0, text="Downloading PDFs...")
    for i, (idx, row) in enumerate(rows):
        url = row.get("attachment_url")
        if isinstance(url, str) and url:
            dest = pdf_path_for(row, idx)
            try:
                download_pdf(url, dest)
                paths[idx] = dest
            except Exception:  # noqa: BLE001
                paths[idx] = None
        else:
            paths[idx] = None
        progress.progress((i + 1) / len(rows), text=f"Downloading PDFs... {i + 1}/{len(rows)}")
    progress.empty()
    return paths


@st.cache_data(show_spinner=False)
def render_pdf_images(path: str, zoom: float = 2.0) -> list:
    """Render every page of a PDF to PNG bytes using PyMuPDF."""
    images = []
    matrix = fitz.Matrix(zoom, zoom)
    with fitz.open(path) as doc:
        for page in doc:
            pix = page.get_pixmap(matrix=matrix)
            images.append(pix.tobytes("png"))
    return images


def render_pdf(path: str) -> None:
    """Display a local PDF as a sequence of page images."""
    try:
        images = render_pdf_images(path)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not render PDF: {exc}")
        return

    if not images:
        st.info("PDF has no pages.")
        return

    for page_no, png in enumerate(images, start=1):
        st.image(png, caption=f"Page {page_no}", use_container_width=True)

    with open(path, "rb") as f:
        st.download_button(
            "Download PDF", data=f.read(), file_name=os.path.basename(path),
            mime="application/pdf",
        )


def main() -> None:
    st.set_page_config(page_title="Predictions & PDFs", layout="wide")
    st.title("June 16 App Predictions Review")

    df = load_data().reset_index(drop=True)
    df["_row_id"] = range(len(df))

    # Download every PDF up front (cached across reruns)
    paths = predownload_all(df)

    # --- Sidebar navigation ---
    st.sidebar.header("Navigation")

    types = ["(all)"] + sorted(df["predicted_types"].dropna().unique().tolist())
    selected_type = st.sidebar.selectbox("Filter by predicted type", types)

    view = df if selected_type == "(all)" else df[df["predicted_types"] == selected_type]
    view = view.reset_index(drop=True)

    if view.empty:
        st.warning("No rows for the selected filter.")
        return

    max_idx = len(view) - 1
    if "idx" not in st.session_state:
        st.session_state.idx = 0
    # Keep the index within range when the filter changes
    st.session_state.idx = min(st.session_state.idx, max_idx)

    def go_prev() -> None:
        st.session_state.idx = max(0, st.session_state.idx - 1)

    def go_next() -> None:
        st.session_state.idx = min(st.session_state.idx + 1, st.session_state.max_idx)

    # Store the current max so the callbacks clamp against the active filter
    st.session_state.max_idx = max_idx

    col_prev, col_mid, col_next = st.sidebar.columns([1, 2, 1])
    col_prev.button("◀ Prev", on_click=go_prev, disabled=st.session_state.idx <= 0)
    col_next.button("Next ▶", on_click=go_next, disabled=st.session_state.idx >= max_idx)

    st.sidebar.number_input(
        "Row", min_value=0, max_value=max_idx, step=1, key="idx"
    )
    st.sidebar.caption(f"Showing row {st.session_state.idx + 1} of {len(view)}")

    row = view.iloc[st.session_state.idx]

    # --- Main layout: predictions on the left, PDF on the right ---
    left, right = st.columns([1, 1.4])

    with left:
        st.subheader("Prediction")
        st.metric("Predicted types", str(row.get("predicted_types", "")))

        meta_cols = [
            "zendesk_id",
            "comment_id",
            "attachment_id",
            "n_predicted_types",
        ]
        meta = {c: row.get(c) for c in meta_cols if c in row.index}
        if meta:
            st.write(meta)

        # Show all non-null prediction-related fields
        pred_fields = row[[c for c in row.index if c not in ("body", "content", "text",
                                                             "text_with_page_markers", "_row_id")]]
        pred_fields = pred_fields.dropna()
        with st.expander("All fields", expanded=False):
            st.dataframe(pred_fields.astype(str).to_frame("value"))

    with right:
        st.subheader("Document")
        path = paths.get(int(row["_row_id"]))
        if not path or not os.path.exists(path):
            url = row.get("attachment_url")
            st.info("No PDF available for this row.")
            if isinstance(url, str) and url:
                st.write(url)
            return
        render_pdf(path)


if __name__ == "__main__":
    main()
