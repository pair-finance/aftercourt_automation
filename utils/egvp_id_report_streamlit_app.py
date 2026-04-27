"""Streamlit app to inspect data for a given EGVP ID.

Run with:
    streamlit run utils/egvp_id_streamlit_app.py
"""
import os
import sys
import tempfile
from datetime import date, timedelta

import boto3
import pandas as pd
import streamlit as st

# Make sibling utils importable when running via `streamlit run`
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from court_team_uploaded_json_utils import search_egvp_id_in_exported_and_archived  # noqa: E402
from prod_utils import get_data_by_egvp_id  # noqa: E402
from python_utilities.db_connection import DbConnection  # noqa: E402


st.set_page_config(page_title="EGVP ID Inspector", layout="wide")


@st.cache_resource(show_spinner="Connecting to analytics DB and AWS...")
def get_clients(aws_profile: str):
    analytics_db = DbConnection("ANALYTICS", "PROD_RDS")
    session = boto3.Session(profile_name=aws_profile)
    s3 = session.client("s3")
    return analytics_db, s3


@st.cache_data(show_spinner="Fetching data by EGVP ID...")
def fetch_egvp_data(egvp_id: str, download_dir: str, aws_profile: str) -> pd.DataFrame:
    analytics_db, s3 = get_clients(aws_profile)
    return get_data_by_egvp_id(
        egvp_id,
        analytics_db,
        s3,
        pdf_download=True,
        pdf_download_dir=download_dir,
        verbose=False,
    )


@st.cache_data(show_spinner="Searching exported & archived S3...")
def fetch_export_archive(egvp_id: str, start_date: date, end_date: date) -> dict:
    return search_egvp_id_in_exported_and_archived(
        egvp_id=egvp_id,
        start_date=start_date,
        end_date=end_date,
    )


def _derive_dates_from_data(data: pd.DataFrame, margin_days: int = 1):
    s3_key = data.iloc[0]["document_s3_key"]
    year, month, day = s3_key.split("/")[1].split("-")
    doc_date = date(int(year), int(month), int(day))
    return (
        doc_date,
        doc_date - timedelta(days=margin_days),
        doc_date + timedelta(days=margin_days),
    )


def main():
    st.title("EGVP ID Inspector")

    with st.sidebar:
        st.header("Settings")
        aws_profile = st.text_input("AWS profile", value="739275445236_DataScienceUser")
        margin_days = st.number_input("Search margin (days)", min_value=0, max_value=30, value=1)
        download_dir = st.text_input(
            "PDF download dir",
            value="/Users/melih.gorgulu/Desktop/Projects/aftercourt_automation/assets/pdfs/tmp/sil",
        )

    egvp_id = st.text_input(
        "EGVP ID",
        value="",
        placeholder="NRW_B217769188566434bfab1a9-518d-4d4b-9b30-fdc76aea19bd",
    )

    if not st.button("Run", type="primary", disabled=not egvp_id):
        st.info("Enter an EGVP ID and press Run.")
        return

    os.makedirs(download_dir, exist_ok=True)

    try:
        data = fetch_egvp_data(egvp_id, download_dir, aws_profile)
    except Exception as exc:
        st.error(f"Failed to fetch data: {exc}")
        return

    if data is None or data.empty:
        st.warning("No data found for this EGVP ID.")
        return

    # ----- Ticket summary -----
    st.subheader("Ticket Summary")
    row0 = data.iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Ticket UUID", str(row0["ticket_uuid"]))
    c2.metric("Status", str(row0["status"]))
    c3.metric("Origin", str(row0["origin"]))
    c4.metric("Attachments", int(data["attachment_id"].nunique()))

    # ----- get_data_by_egvp_id results -----
    st.subheader("get_data_by_egvp_id results")
    st.dataframe(data, use_container_width=True)

    # ----- Per-attachment view: PDF + textract text + predictions -----
    st.subheader("Attachments")
    unique_attachments = data["attachment_id"].unique()

    for idx, attch in enumerate(unique_attachments, 1):
        sub = data[data["attachment_id"] == attch]
        file_name = sub["file_name"].iloc[0]
        with st.expander(f"Attachment {idx}/{len(unique_attachments)} — {file_name}", expanded=(idx == 1)):
            pdf_path = os.path.join(download_dir, f"{egvp_id}_{attch}.pdf")
            left, right = st.columns([3, 2])

            with left:
                st.markdown("**PDF**")
                if os.path.exists(pdf_path):
                    with open(pdf_path, "rb") as f:
                        pdf_bytes = f.read()
                    st.download_button(
                        "Download PDF",
                        data=pdf_bytes,
                        file_name=os.path.basename(pdf_path),
                        mime="application/pdf",
                        key=f"dl_{attch}",
                    )
                    # Inline preview via embedded PDF viewer
                    import base64
                    b64 = base64.b64encode(pdf_bytes).decode("utf-8")
                    st.markdown(
                        f'<iframe src="data:application/pdf;base64,{b64}" '
                        f'width="100%" height="800" type="application/pdf"></iframe>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.warning(f"PDF not found at {pdf_path}")

            with right:
                st.markdown("**Predictions**")
                preds = sub[["model_name", "type", "subtype", "value"]].drop_duplicates()
                st.dataframe(preds, use_container_width=True, hide_index=True)

                st.markdown("**Textract text**")
                text = sub["text"].iloc[0] if "text" in sub.columns else ""
                st.text_area("Text", value=text or "", height=400, key=f"txt_{attch}")

    # ----- Export / Archive search -----
    st.subheader("Exported & Archived content")
    try:
        doc_date, start_d, end_d = _derive_dates_from_data(data, margin_days=int(margin_days))
        st.caption(f"Document date: {doc_date} — searching {start_d} → {end_d}")
        result = fetch_export_archive(egvp_id, start_d, end_d)
    except Exception as exc:
        st.error(f"Export/Archive search failed: {exc}")
        return

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            f"**Exported:** {'✅ found' if result['found_in_exported'] else '❌ not found'}"
        )
        if result["exported_content"] is not None:
            st.json(result["exported_content"])
    with c2:
        st.markdown(
            f"**Archived:** {'✅ found' if result['found_in_archived'] else '❌ not found'}"
        )
        if result["archived_content"] is not None:
            st.json(result["archived_content"])


if __name__ == "__main__":
    main()
