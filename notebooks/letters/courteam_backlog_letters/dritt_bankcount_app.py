"""Streamlit app to review the 50 Drittauskunft / bank-count documents.

Left column  : OCR text with page markers (``text_with_page_markers``)
Right column : the original PDF, downloaded on the fly via the
               ``attachment_url`` token and rendered inline.

Run with:
    streamlit run dritt_bankcount_app.py
"""

from __future__ import annotations

import ast
import base64
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

CSV_PATH = Path(__file__).with_name("dritt_bankcount_streamlit.csv")
REQUEST_TIMEOUT = 30


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    return pd.read_csv(CSV_PATH)


@st.cache_data(show_spinner="Downloading document…")
def download_pdf(url: str) -> bytes:
    response = requests.get(url, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.content


def parse_banks(raw) -> list[str]:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return []
    if isinstance(raw, list):
        return [str(x) for x in raw]
    try:
        value = ast.literal_eval(str(raw))
        if isinstance(value, (list, tuple)):
            return [str(x) for x in value]
        return [str(value)]
    except (ValueError, SyntaxError):
        return [str(raw)]


def render_pdf(pdf_bytes: bytes, height: int = 900) -> None:
    base64_pdf = base64.b64encode(pdf_bytes).decode("utf-8")
    st.markdown(
        f"""
        <iframe
            src="data:application/pdf;base64,{base64_pdf}"
            width="100%"
            height="{height}"
            style="border: 1px solid #ddd; border-radius: 6px;"
            type="application/pdf">
        </iframe>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title="Dritt / Bank-count document review", layout="wide")

    df = load_data()
    total = len(df)

    st.title("Drittauskunft / Bank-count document review")

    # --- Navigation -----------------------------------------------------
    if "idx" not in st.session_state:
        st.session_state.idx = 0

    top = st.columns([1, 1, 6, 2])
    with top[0]:
        if st.button("⬅️ Prev", use_container_width=True):
            st.session_state.idx = (st.session_state.idx - 1) % total
    with top[1]:
        if st.button("Next ➡️", use_container_width=True):
            st.session_state.idx = (st.session_state.idx + 1) % total
    with top[2]:
        st.session_state.idx = st.slider(
            "Document", 0, total - 1, st.session_state.idx, label_visibility="collapsed"
        )
    with top[3]:
        st.markdown(f"**{st.session_state.idx + 1} / {total}**")

    row = df.iloc[st.session_state.idx]
    banks = parse_banks(row.get("extracted_banks"))
    n_banks = row.get("dritt_n_banks")

    # --- Bank-count summary --------------------------------------------
    meta = st.columns([1, 1, 2, 2])
    meta[0].metric(
        "Bank accounts",
        "—" if pd.isna(n_banks) else int(n_banks),
    )
    meta[1].metric("Matches found", len(banks))
    if "zendesk_id" in row:
        meta[2].metric("zendesk_id", str(row["zendesk_id"]))
    if "attachment_id" in row:
        meta[3].metric("attachment_id", str(row["attachment_id"]))

    st.subheader("Extracted bank matches")
    if banks:
        st.dataframe(
            pd.DataFrame({"#": range(1, len(banks) + 1), "match": banks}),
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.info("No bank matches extracted for this document.")

    st.divider()

    # --- Side-by-side ---------------------------------------------------
    left, right = st.columns(2)

    with left:
        st.subheader("Parsed text (with page markers)")
        text_value = ""
        for col in ("text_with_page_markers", "text"):
            value = row.get(col)
            if value is not None and not (isinstance(value, float) and pd.isna(value)):
                text_value = str(value)
                if text_value.strip():
                    break
        st.text_area(
            "text_with_page_markers",
            value=text_value,
            height=900,
            label_visibility="collapsed",
        )

    with right:
        st.subheader("Original document")
        url = ""
        for col in ("attachment_url", "attachment_url_x", "attachment_url_y"):
            value = str(row.get(col, "")).strip()
            if value and value.lower() != "nan":
                url = value
                break
        if not url:
            st.warning("No attachment_url available for this document.")
        else:
            st.markdown(f"[Open in new tab]({url})")
            try:
                pdf_bytes = download_pdf(url)
                render_pdf(pdf_bytes)
                st.download_button(
                    "Download PDF",
                    data=pdf_bytes,
                    file_name=f"{row.get('attachment_id', st.session_state.idx)}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
            except requests.RequestException as exc:
                st.error(f"Failed to download the document: {exc}")


if __name__ == "__main__":
    main()
