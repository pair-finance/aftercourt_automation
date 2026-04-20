import streamlit as st
import pandas as pd
import os
from pdf2image import convert_from_path

DATA_PATH = "/Users/melih.gorgulu/Desktop/Projects/aftercourt_automation/data/raw/final_raw_data.csv"
PDF_DIR = "/Users/melih.gorgulu/Desktop/Projects/aftercourt_automation/assets/pdfs/tmp/possible_vormegenverzeicnes"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    # extract filename from object_key
    df["pdf_filename"] = df["object_key"].apply(
        lambda x: x.split("/")[-1] if pd.notna(x) else None
    )
    # filter to rows whose PDF exists in the directory
    available_pdfs = set(os.listdir(PDF_DIR))
    df = df[df["pdf_filename"].isin(available_pdfs)].reset_index(drop=True)
    return df


@st.cache_data
def get_pdf_images(filepath):
    return convert_from_path(filepath, dpi=100)


def show_pdf(filepath):
    images = get_pdf_images(filepath)
    total_pages = len(images)

    if "page_idx" not in st.session_state:
        st.session_state.page_idx = 0

    page_idx = st.session_state.page_idx
    if page_idx >= total_pages:
        st.session_state.page_idx = 0
        page_idx = 0

    _, center, _ = st.columns([1, 2, 1])
    with center:
        st.image(images[page_idx], width=600)
        p1, p2, p3 = st.columns([1, 2, 1])
        with p1:
            if st.button("⬅ Prev Page", disabled=page_idx == 0):
                st.session_state.page_idx -= 1
                st.rerun()
        with p2:
            st.markdown(f"**Page {page_idx + 1} / {total_pages}**")
        with p3:
            if st.button("Next Page ➡", disabled=page_idx >= total_pages - 1):
                st.session_state.page_idx += 1
                st.rerun()


def append_index_to_file(filename, index):
    filepath = os.path.join(SCRIPT_DIR, filename)
    with open(filepath, "a") as f:
        f.write(f"{index}\n")


def main():
    st.set_page_config(page_title="Document Viewer", layout="wide")
    st.markdown(
        """<style>
        .block-container { padding-top: 1rem; padding-bottom: 0rem; }
        h1 { margin-bottom: 0; font-size: 1.5rem; }
        [data-testid="stMetric"] { padding: 0; }
        [data-testid="stMetricValue"] { font-size: 1rem; }
        </style>""",
        unsafe_allow_html=True,
    )
    st.title("Document Viewer")

    df = load_data()

    if len(df) == 0:
        st.error("No matching documents found.")
        return

    if "idx" not in st.session_state:
        st.session_state.idx = 0

    idx = st.session_state.idx
    row = df.iloc[idx]

    # --- Left: controls | Right: PDF ---
    left, right = st.columns([1.5, 2])

    with left:
        # Navigation
        st.markdown(f"**Document {idx + 1} / {len(df)}**")
        if st.button("⬅ Previous", disabled=idx == 0):
            st.session_state.idx -= 1
            st.session_state.page_idx = 0
            st.rerun()

        # Info panel
        st.metric("Document Type", row["document_type"])
        st.metric("is_pfub", row["is_pfub"])
        st.metric("is_ladung", row["is_ladung"])

        # Labeling buttons
        ticket_uuid = row["ticket_uuid"]

        def _label_and_next(filename):
            append_index_to_file(filename, ticket_uuid)
            if idx < len(df) - 1:
                st.session_state.idx += 1
                st.session_state.page_idx = 0

        if st.button("🚫 Its NOT LADUNG DOC"):
            _label_and_next("not_ladung.txt")
            st.rerun()
        if st.button("✅ Its VE DOC"):
            _label_and_next("ve.txt")
            st.rerun()
        if st.button("✅ Labeling is correct"):
            _label_and_next("correct.txt")
            st.rerun()

    with right:
        pdf_path = os.path.join(PDF_DIR, row["pdf_filename"])
        images = get_pdf_images(pdf_path)
        total_pages = len(images)

        if "page_idx" not in st.session_state:
            st.session_state.page_idx = 0
        page_idx = st.session_state.page_idx
        if page_idx >= total_pages:
            st.session_state.page_idx = 0
            page_idx = 0

        p1, p2, p3 = st.columns([1, 2, 1])
        with p1:
            if st.button("⬅ Prev Page", disabled=page_idx == 0):
                st.session_state.page_idx -= 1
                st.rerun()
        with p2:
            st.markdown(f"**Page {page_idx + 1} / {total_pages}**")
        with p3:
            if st.button("Next Page ➡", disabled=page_idx >= total_pages - 1):
                st.session_state.page_idx += 1
                st.rerun()

        st.image(images[page_idx], use_container_width=True)


if __name__ == "__main__":
    main()
