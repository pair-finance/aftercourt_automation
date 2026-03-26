import streamlit as st
import pandas as pd
import os
import base64
import re
import json

st.set_page_config(layout="wide", page_title="Wrong Predictions Review")

PDF_DIR = "/Users/melih.gorgulu/Desktop/Projects/aftercourt_automation/assets/pdfs/tmp/WRONG_PREDICTIONS_CHECK_REASONING"
DATA_PATH = "/Users/melih.gorgulu/Desktop/Projects/aftercourt_automation/notebooks/analysis_data/final_preds_pfub_invoice_new_propmt.csv"

WRONG_IDS = [53752407, 50426386, 52426004, 51179979, 52249553, 52426088, 51469512, 52885140, 52885159, 53790961, 53408148, 53015376, 53408147, 52885159, 50426389]


def extract_first_json(text):
    text = (
        text.replace(" ", "").replace("\n", "")
        .replace("True", "true").replace("False", "false")
        .replace("None", "null").replace("'", '"')
    )
    match = re.search(r'\{.*?\}', text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            return {}
    return {}


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df[df["attachment_id"].isin(WRONG_IDS)].copy()
    df["pred_json"] = df["pred_llm"].apply(extract_first_json)
    df["llm_is_pfub"] = df["pred_json"].apply(lambda x: x.get("is_pfub", None))
    df["llm_is_invoice"] = df["pred_json"].apply(lambda x: x.get("is_invoice_inside", None))

    # Try to compute invoice_token_count if clean_text exists
    if "invoice_token_count" not in df.columns and "clean_text" in df.columns:
        tokens = ['rechnung', 'erinnerung', 'zahlung', 'chemnitz', 'landesjustizkasse',
                  'betrag', 'landeshauptkasse', 'kassenzeichen', 'abs', 'kostenrechnung']
        pattern = re.compile(r'\b(' + '|'.join(re.escape(t) for t in tokens) + r')\b')
        df["invoice_token_count"] = df["clean_text"].apply(lambda t: len(pattern.findall(str(t).lower())))

    return df


def find_pdf(attachment_id):
    for f in os.listdir(PDF_DIR):
        if f.startswith(str(attachment_id)) and f.endswith(".pdf"):
            return os.path.join(PDF_DIR, f)
    return None


def render_pdf(pdf_path):
    with open(pdf_path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    return f'<iframe src="data:application/pdf;base64,{data}" width="100%" height="800" type="application/pdf"></iframe>'


data = load_data()

st.title("Wrong Predictions Review")

attachment_ids = data["attachment_id"].unique().tolist()
idx = st.sidebar.selectbox("Select row index", range(len(attachment_ids)), format_func=lambda i: f"{i+1}. attachment {attachment_ids[i]}")
attachment_id = attachment_ids[idx]
row = data[data["attachment_id"] == attachment_id].iloc[0]

left, right = st.columns([1, 1])

with left:
    st.subheader(f"PDF — attachment {attachment_id}")
    pdf_path = find_pdf(attachment_id)
    if pdf_path:
        st.markdown(render_pdf(pdf_path), unsafe_allow_html=True)
    else:
        st.error(f"PDF not found for attachment_id: {attachment_id}")

with right:
    st.subheader("Prediction Results")

    cols = st.columns(4)
    cols[0].metric("pfub_prob", f"{row.get('pfub_prob', 'N/A'):.4f}" if pd.notna(row.get('pfub_prob')) else "N/A")
    cols[1].metric("llm_is_pfub", str(row.get("llm_is_pfub", "N/A")))
    cols[2].metric("llm_is_invoice", str(row.get("llm_is_invoice", "N/A")))
    cols[3].metric("invoice_token_count", str(row.get("invoice_token_count", "N/A")))

    st.divider()
    st.subheader("LLM Reasoning (pred_llm)")
    st.text_area("", value=str(row.get("pred_llm", "")), height=600, disabled=True)
