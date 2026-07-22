"""Streamlit app to inspect data for a given EGVP ID.

Run with:
    streamlit run utils/egvp_id_streamlit_app.py
"""
import os
import sys
import json
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


@st.cache_data(show_spinner="Fetching final intents from egvp_intents...")
def fetch_egvp_intents(attachment_ids: tuple, aws_profile: str) -> dict:
    """Map attachment id -> the final intent entry sent to the EGVP side.

    Reads the ``egvp_intents`` table. The ``intents`` column holds a JSON list
    of per-attachment entries, each with a flat ``attachment_id``,
    ``aftercourt_type`` (the final intent), ``is_aftercourt``,
    ``full_auto_confidence`` and extracted ``params``. For each attachment we
    read its row and pick the entry whose ``attachment_id`` matches.
    """
    intent_map: dict = {}
    if not attachment_ids:
        return intent_map

    analytics_db, _ = get_clients(aws_profile)
    ids_sql = ", ".join(f"'{a}'" for a in attachment_ids)
    query = f"""
        SELECT attachment_id, intents, ready_for_automation, created_at
        FROM egvp_intents
        WHERE attachment_id IN ({ids_sql})
        ORDER BY created_at DESC
    """
    df = analytics_db.sql_to_df(query)

    for _, row in df.iterrows():
        aid = str(row["attachment_id"])
        if aid in intent_map:
            continue  # rows are ordered newest first; keep the latest
        intents_raw = row["intents"]
        try:
            intents = json.loads(intents_raw) if isinstance(intents_raw, str) else intents_raw
        except (TypeError, ValueError):
            continue
        if not intents:
            continue
        entry = next(
            (e for e in intents if str(e.get("attachment_id")) == aid),
            intents[0],
        )
        entry = dict(entry)
        entry["ready_for_automation"] = row.get("ready_for_automation")
        intent_map[aid] = entry

    return intent_map


def _derive_dates_from_data(data: pd.DataFrame, margin_days: int = 1):
    s3_key = data.iloc[0]["document_s3_key"]
    year, month, day = s3_key.split("/")[1].split("-")
    doc_date = date(int(year), int(month), int(day))
    return (
        doc_date,
        doc_date - timedelta(days=margin_days),
        doc_date + timedelta(days=margin_days),
    )


def _is_true(value) -> bool:
    """Normalise the stringified booleans stored in llm_attachments_predictions.

    Values may be stored as ``True``, ``'True'`` or even quoted like ``"'True'"``.
    """
    if isinstance(value, str):
        value = value.strip().strip("'\"").strip().lower()
    return value in (True, 1, "true", "1", "yes")


# The per-model signals we care about, in display order: (model type, subtype).
KEY_PREDICTIONS = [
    ("vermogenverzeichnis_egvp", "is_va"),
    ("drittauskunft_egvp", "is_dritt"),
    ("pfub_erlass_egvp", "is_pfub"),
    ("pfub_erlass_egvp", "is_invoice_inside"),
    ("invoice_detection_egvp", "is_invoice_inside"),
    ("invoice_detection_egvp", "start_page"),
    ("invoice_detection_egvp", "end_page"),
    ("egvp_standalone_invoice", "invoice"),
]


def _clean_value(value):
    if isinstance(value, str):
        return value.strip().strip("'\"").strip()
    return value


def _key_predictions(sub: pd.DataFrame) -> pd.DataFrame:
    """Look up the key per-model signals so you can see which model says what.

    Returns a table with one row per (model, field) in ``KEY_PREDICTIONS``,
    matching on either the ``type`` or ``model_name`` column. Missing signals
    show ``—``.
    """
    rows = []
    for model, subtype in KEY_PREDICTIONS:
        match = sub[
            ((sub["type"] == model) | (sub["model_name"] == model)) & (sub["subtype"] == subtype)
        ]
        value = _clean_value(match["value"].iloc[0]) if not match.empty else "—"
        rows.append({"model": model, "field": subtype, "value": value})
    return pd.DataFrame(rows, columns=["model", "field", "value"])


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

    if st.button("Run", type="primary", disabled=not egvp_id):
        st.session_state["run_egvp_id"] = egvp_id

    egvp_id = st.session_state.get("run_egvp_id")
    if not egvp_id:
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

    # ----- Final intents (from egvp_intents DB, matched by attachment id) -----
    intent_map: dict = {}
    try:
        attachment_ids = tuple(str(a) for a in data["attachment_id"].unique())
        intent_map = fetch_egvp_intents(attachment_ids, aws_profile)
    except Exception as exc:
        st.warning(f"Fetching final intents failed: {exc}")

    # ----- Export / Archive search (raw content) -----
    export_result = None
    try:
        doc_date, start_d, end_d = _derive_dates_from_data(data, margin_days=int(margin_days))
        export_result = fetch_export_archive(egvp_id, start_d, end_d)
    except Exception as exc:
        st.warning(f"Export/Archive search failed: {exc}")

    # ----- Per-attachment view: predictions + intent + PDF + text -----
    st.subheader("Attachments")
    unique_attachments = data["attachment_id"].unique()

    for idx, attch in enumerate(unique_attachments, 1):
        sub = data[data["attachment_id"] == attch]
        file_name = sub["file_name"].iloc[0]
        intent_entry = intent_map.get(str(attch))
        intent_label = (intent_entry or {}).get("aftercourt_type", "—")

        with st.expander(
            f"Attachment {idx}/{len(unique_attachments)} — {file_name}  ·  intent: {intent_label}",
            expanded=(idx == 1),
        ):
            # ----- Key per-model predictions (which model says what) -----
            st.markdown("**Key model predictions**")
            st.dataframe(
                _key_predictions(sub),
                use_container_width=True,
                hide_index=True,
            )

            # ----- Final intent sent to EGVP -----
            st.markdown("**Final intent sent to EGVP**")
            if intent_entry is None:
                st.info("No final intent found for this attachment in egvp_intents.")
            else:
                is_aftercourt = intent_entry.get("is_aftercourt", False)
                full_auto = intent_entry.get("full_auto_confidence", False)
                ready = intent_entry.get("ready_for_automation")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Intent", str(intent_entry.get("aftercourt_type", "unknown")))
                m2.metric("Is aftercourt", "✅ yes" if is_aftercourt else "❌ no")
                m3.metric("Full auto", "✅ yes" if full_auto else "❌ no")
                m4.metric("Ready for automation", "✅ yes" if ready else "❌ no")
                params = intent_entry.get("params", {}) or {}
                if params:
                    st.markdown("**Extracted params**")
                    st.json(params, expanded=False)

            # ----- Tabs: PDF | Predictions | Text (keeps the page compact) -----
            tab_pdf, tab_preds, tab_text = st.tabs(["📄 PDF", "🤖 Predictions", "📝 Text"])

            with tab_pdf:
                pdf_path = os.path.join(download_dir, f"{egvp_id}_{attch}.pdf")
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
                    import base64
                    b64 = base64.b64encode(pdf_bytes).decode("utf-8")
                    st.markdown(
                        f'<iframe src="data:application/pdf;base64,{b64}" '
                        f'width="100%" height="600" type="application/pdf"></iframe>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.warning(f"PDF not found at {pdf_path}")

            with tab_preds:
                preds = sub[["model_name", "type", "subtype", "value"]].drop_duplicates()
                st.dataframe(preds, use_container_width=True, hide_index=True)

            with tab_text:
                text = sub["text"].iloc[0] if "text" in sub.columns else ""
                st.text_area("Textract text", value=text or "", height=400, key=f"txt_{attch}")

    # ----- Export / Archive search (raw content) -----
    st.subheader("Exported & Archived content")
    if export_result is None:
        st.error("Export/Archive search was not available for this EGVP ID.")
        return

    result = export_result
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
