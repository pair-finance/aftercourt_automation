import os

import fitz  # PyMuPDF
import pandas as pd
import streamlit as st

CSV_PATH = "/Users/melih.gorgulu/Desktop/Projects/aftercourt_automation/notebooks/letters/final_letter_data_all_preds.csv"
PDF_DIR = "/Users/melih.gorgulu/Desktop/Projects/aftercourt_automation/assets/letters"

st.set_page_config(page_title="Aftercourt Predictions", layout="wide")


@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH)
    df["is_va"] = df["is_va"].fillna(False).astype(bool)
    df["is_dritt"] = df["is_dritt"].fillna(False).astype(bool)
    return df


def pdf_path(attachment_id: str) -> str:
    return os.path.join(PDF_DIR, f"{attachment_id}.pdf")


@st.cache_data(show_spinner=False)
def render_pdf_pages(path: str, zoom: float = 2.0) -> list[bytes]:
    """Render each PDF page to a PNG image (bytes)."""
    images = []
    matrix = fitz.Matrix(zoom, zoom)
    with fitz.open(path) as doc:
        for page in doc:
            pix = page.get_pixmap(matrix=matrix)
            images.append(pix.tobytes("png"))
    return images


def show_pdf(path: str) -> None:
    pages = render_pdf_pages(path)
    for i, img in enumerate(pages, start=1):
        st.image(img, caption=f"Page {i}", use_container_width=True)


def main() -> None:
    df = load_data()

    st.sidebar.header("Filters")
    filter_mode = st.sidebar.radio(
        "Show predictions where",
        ("is_va AND is_dritt", "is_va OR is_dritt", "is_va only", "is_dritt only"),
    )

    if filter_mode == "is_va AND is_dritt":
        mask = df["is_va"] & df["is_dritt"]
    elif filter_mode == "is_va OR is_dritt":
        mask = df["is_va"] | df["is_dritt"]
    elif filter_mode == "is_va only":
        mask = df["is_va"]
    else:
        mask = df["is_dritt"]

    filtered = df[mask].reset_index(drop=True)

    st.title("Aftercourt Predictions Viewer")
    st.write(f"**{len(filtered)}** attachments match `{filter_mode}`.")

    if filtered.empty:
        st.info("No attachments match the selected filter.")
        return

    options = filtered["attachment_id"].tolist()
    selected_id = st.sidebar.selectbox("Attachment", options)

    row = filtered[filtered["attachment_id"] == selected_id].iloc[0]

    col_meta, col_pdf = st.columns([1, 2])

    with col_meta:
        st.subheader("Metadata")
        st.write(f"**Attachment ID:** {row['attachment_id']}")
        if "zendesk_id" in row:
            st.write(f"**Zendesk ID:** {row['zendesk_id']}")
        st.write(f"**is_va:** {row['is_va']}")
        st.write(f"**is_dritt:** {row['is_dritt']}")
        if "pfub_erlass_prob" in row:
            st.write(f"**pfub_erlass_prob:** {row['pfub_erlass_prob']:.3f}")
        if "ladung_va_prob" in row:
            st.write(f"**ladung_va_prob:** {row['ladung_va_prob']:.3f}")
        with st.expander("Extracted text"):
            st.text(str(row.get("text", "")))

    with col_pdf:
        st.subheader("PDF")
        path = pdf_path(selected_id)
        if os.path.exists(path):
            show_pdf(path)
            with open(path, "rb") as f:
                st.download_button(
                    "Download PDF", f, file_name=f"{selected_id}.pdf", mime="application/pdf"
                )
        else:
            st.warning(f"PDF not found: {path}")


if __name__ == "__main__":
    main()
