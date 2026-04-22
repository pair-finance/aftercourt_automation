"""
Streamlit app for labeling PDFs as PFUB or NOT PFUB.

Usage:
    streamlit run pfub_labeler.py
"""

import os
import io
import pandas as pd
import streamlit as st
import fitz  # pymupdf

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "add_data_sample_sil.csv")
PDF_DIR = os.path.join(BASE_DIR, "..", "..", "..", "assets", "pdfs", "tmp", "add_data_sil")
LABELS_FILE = os.path.join(BASE_DIR, "pfub_labels.txt")

# --- Load data ---
@st.cache_data
def load_data():
    df = pd.read_csv(CSV_PATH)
    return df

def load_labeled_ids():
    if os.path.exists(LABELS_FILE):
        with open(LABELS_FILE, "r") as f:
            return set(line.strip() for line in f if line.strip())
    return set()

df = load_data()

# Filter out already-labeled attachment IDs
labeled_ids = load_labeled_ids()
df = df[~df["attachment_id"].astype(str).isin(labeled_ids)].reset_index(drop=True)

# --- Session state ---
if "current_idx" not in st.session_state:
    st.session_state.current_idx = 0

idx = st.session_state.current_idx

if idx >= len(df):
    st.success("All PDFs have been reviewed!")
    st.stop()

row = df.iloc[idx]
attachment_id = str(row["attachment_id"])
pfub_prob = row["pfub_prob"]
pred_is_pfub = row["pred_is_pfub"]
pred_is_invoice_inside = row["pred_is_invoice_inside"]
pdf_path = os.path.join(PDF_DIR, f"{attachment_id}.pdf")

# --- Header info ---
st.title("PFUB Labeler")
st.markdown(f"**PDF {idx + 1} / {len(df)}**")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Attachment ID", attachment_id)
col2.metric("pfub_prob", f"{pfub_prob:.4f}")
col3.metric("pred_is_pfub", str(pred_is_pfub))
col4.metric("pred_is_invoice_inside", str(pred_is_invoice_inside))

# --- Buttons ---
btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 2])

with btn_col1:
    pfub_clicked = st.button("✅ PFUB", type="primary", use_container_width=True)
with btn_col2:
    not_pfub_clicked = st.button("❌ NOT PFUB", type="secondary", use_container_width=True)
with btn_col3:
    st.empty()

if pfub_clicked:
    with open(LABELS_FILE, "a") as f:
        f.write(attachment_id + "\n")
    st.session_state.current_idx += 1
    st.rerun()

if not_pfub_clicked:
    st.session_state.current_idx += 1
    st.rerun()

# --- Render PDF as images ---
if os.path.exists(pdf_path):
    doc = fitz.open(pdf_path)
    for page_num in range(len(doc)):
        page = doc[page_num]
        pix = page.get_pixmap(dpi=200)
        img_bytes = pix.tobytes("png")
        st.image(img_bytes, caption=f"Page {page_num + 1}", use_container_width=True)
    doc.close()
else:
    st.error(f"PDF not found: {pdf_path}")

# --- Style the buttons ---
st.markdown(
    """
    <style>
    div[data-testid="column"]:nth-child(1) button {
        background-color: #28a745 !important;
        color: white !important;
        border: none !important;
    }
    div[data-testid="column"]:nth-child(2) button {
        background-color: #dc3545 !important;
        color: white !important;
        border: none !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
