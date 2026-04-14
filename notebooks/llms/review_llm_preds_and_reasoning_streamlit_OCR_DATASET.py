import streamlit as st
import pandas as pd
import os
import base64
import re
import json

st.set_page_config(layout="wide", page_title="Wrong Predictions Review")

st.markdown("""
<style>
/* Green "LLM: Correct" button */
button[kind="primary"] {
    background-color: #28a745 !important;
    border-color: #28a745 !important;
    color: white !important;
}
/* Red "LLM: Wrong" button */
button[kind="secondary"] {
    background-color: #dc3545 !important;
    border-color: #dc3545 !important;
    color: white !important;
}
</style>
""", unsafe_allow_html=True)

PDF_DIR = "/Users/melih.gorgulu/Desktop/Projects/aftercourt_automation/assets/pdfs/tmp/ocr/fps"
DATA_PATH = "/Users/melih.gorgulu/Desktop/Projects/aftercourt_automation/notebooks/analysis_data/llm_predictions/ocr_raw_data_plus_prod_2025-01-20_to_2025-02-27_PREDS_31_03_2026_PROMPT_V3.csv"

WRONG_ATTACHMENT_IDS = ['8eb7ea40-2bb6-5828-98af-5c8686c67a34',
 'ff5c534d-1dd9-5dac-be2e-b57f812f36ac',
 '3faf342b-f569-586f-9c82-51a8622ec01f',
 '10a85105-5450-5237-9d56-d3ff8be27225',
 '655a9aa2-ee49-592e-ab27-6b54893cc058',
 'dc7765b9-2583-5b20-9b9e-17d716ce515c',
 '82675ccc-691c-59ec-b4c7-da6abb479c02',
 'cb52ff80-955e-5641-97ab-7f96e18b2e0d',
 '96035b43-7b34-5c41-a7ae-0799d634e6dd',
 'e9b9507f-518b-59cf-9a30-d557205315af',
 '969a047a-4013-51f9-9090-682e16213da1',
 'a900bd2b-bacd-52a5-9ec5-d040dbbd65a5',
 '8ce044ba-cb50-562e-9a6a-fbba9507b39a',
 'c18df4fc-aec9-5af3-87c9-04686ab1cf9f',
 '0c49b2d5-5441-555e-ae8e-d93d7fcf7a31',
 'acf0f7d7-f777-521c-935d-e770754b046d',
 'a7f979e2-aaa9-5903-b078-83b24fbf5d25',
 '0e36359a-398d-5446-8833-851d7ca4236d',
 '309dba0c-01a8-5375-b573-4d9b9608fe18',
 '5ce7aff4-5796-5a49-83b4-78c3596e1ad0',
 '1c757976-4db4-5708-aa61-ded76a17b3c2',
 '3fd5e686-3b0f-5221-9d30-7db21cb162dd',
 '60c33a90-a9eb-511d-bf78-9a8a0defd562',
 'e205ef3e-3956-5038-a798-2313a00b8193',
 '39e7b443-3c8e-5161-be90-b92af91ff7eb',
 '8fd9d839-49fb-5801-97ac-2bac72023db8',
 'e2a2e1b1-81ff-5efe-bf1e-24ba44b7bd1e',
 'bff94d5c-03ab-5058-8b6e-505aa4a18d38',
 '38a55d63-4252-53ac-a74c-6044d2e42e85',
 '616449bf-357b-5f52-86bd-d767f6028645',
 'b602f2f1-8150-5dde-a3a9-25828d058cf4',
 '7297decb-0830-503a-aa55-104d08e0e798',
 '38ca6e50-bb2a-5352-b1b9-0c6d70c7783d',
 '217dbf81-7f27-5dfe-b4ac-660e7b49e29c',
 '6eeeb51a-fd2d-5054-baff-f904edf2fe38',
 'a3881177-4d9f-546a-aca8-a6684438fedd',
 '7745cd44-4b8e-574e-9a55-789526aa21c7',
 'bea88144-79ee-58c2-9a23-57dd60dbdf8a',
 '72c09e6c-ae34-58db-a9fe-a2e7ee8eceeb',
 '7bdb895f-643c-58a5-bf71-098f4e0d4a4d']



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
    df = df[df["attachment_id"].isin(WRONG_ATTACHMENT_IDS)].copy()
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


def find_pdf(object_key):
    name = object_key.split("/")[-1]
    pdf_filename = f"{name}"
    pdf_path = os.path.join(PDF_DIR, pdf_filename)
    if os.path.exists(pdf_path):
        return pdf_path
    return None


def render_pdf(pdf_path):
    with open(pdf_path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    return f'<iframe src="data:application/pdf;base64,{data}" width="100%" height="800" type="application/pdf"></iframe>'


data = load_data()

st.title("Wrong Predictions Review")

attachment_ids = data["attachment_id"].unique().tolist()

# Apply pending advance before widget renders
if "advance_doc" in st.session_state:
    st.session_state["doc_idx"] = st.session_state.pop("advance_doc")

idx = st.sidebar.selectbox("Select row index", range(len(attachment_ids)), format_func=lambda i: f"{i+1}. attachment {attachment_ids[i]}", key="doc_idx")
attachment_id = attachment_ids[idx]
row = data[data["attachment_id"] == attachment_id].iloc[0]

left, right = st.columns([1, 1])

with left:
    object_key = row.get('object_key', None)
    if pd.isna(object_key):
        object_key = attachment_id
    display_name = str(object_key).split("/")[-1]
    st.subheader(f"PDF — {display_name}")
    pdf_path = find_pdf(str(object_key))
    if pdf_path:
        file_size_mb = os.path.getsize(pdf_path) / (1024 * 1024)
        if file_size_mb > 2:
            st.warning(f"PDF is large ({file_size_mb:.1f} MB), rendering may be slow.")
        st.markdown(render_pdf(pdf_path), unsafe_allow_html=True)
    else:
        searched_name = str(object_key).split("/")[-1]
        searched_path = os.path.join(PDF_DIR, searched_name)
        st.error(f"PDF not found. Searched path: {searched_path}")

with right:
    st.subheader("Prediction Results")

    st.markdown(
        '<style>div[data-testid="stMetric"] label {font-size: 0.75rem;} div[data-testid="stMetric"] div[data-testid="stMetricValue"] {font-size: 0.9rem;}</style>',
        unsafe_allow_html=True,
    )
    cols = st.columns(5)
    cols[0].metric("is_pfub", str(row.get("is_pfub", "N/A")))
    cols[1].metric("llm_is_pfub", str(row.get("llm_is_pfub", "N/A")))
    cols[2].metric("llm_is_invoice", str(row.get("llm_is_invoice", "N/A")))
    cols[3].metric("invoice_token_count", str(row.get("invoice_token_count", "N/A")))
    cols[4].metric("document_type", str(row.get("document_type", "N/A")))

    st.divider()

    TXT_DIR = "/Users/melih.gorgulu/Desktop/Projects/aftercourt_automation/notebooks/llms"
    btn_cols = st.columns(2)
    with btn_cols[0]:
        if st.button("LLM: Correct", type="primary", key="btn_correct"):
            path = os.path.join(TXT_DIR, "ocr_pfub_fixes.txt")
            with open(path, "a") as f:
                f.write(f"{attachment_id}\n")
            st.session_state["advance_doc"] = min(idx + 1, len(attachment_ids) - 1)
            st.rerun()
    with btn_cols[1]:
        if st.button("LLM: Wrong", type="secondary", key="btn_wrong"):
            path = os.path.join(TXT_DIR, "llm_pred_wrong.txt")
            with open(path, "a") as f:
                f.write(f"{attachment_id}\n")
            st.session_state["advance_doc"] = min(idx + 1, len(attachment_ids) - 1)
            st.rerun()

    st.subheader("LLM Reasoning (pred_llm)")
    st.text_area("", value=str(row.get("pred_llm", "")), height=600, disabled=True)
