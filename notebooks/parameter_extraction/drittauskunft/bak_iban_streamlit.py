"""Streamlit app to inspect the BAK -> IBAN extraction results.

Left side  : the OCR document text with BAK numbers and their IBANs highlighted.
             Each BAK and the IBANs that belong to it share the same color.
Right side : the local PDF of the document.
Top        : the `no_result`, `bank_numbers` and `bak_to_ibans` column values.

Run with:
    streamlit run bak_iban_streamlit.py
"""

import ast
import base64
import html
import os
import re
from collections import deque

import pandas as pd
import streamlit as st

CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dritt_streamlit_latestiban.csv")

# Same patterns used by the extraction algorithm, needed here only to locate spans.
DRITTAUSKUNFT_BAK_NUMBER_REGEX = re.compile(
    r"\bbak\s*-\s*(?:nr|nummer)\.?:?\s*\d{6}\b", re.IGNORECASE
)
KONTOINHABER_REGEX = re.compile(r"\bkontoinhaber\b", re.IGNORECASE)
KONTONUMMER_REGEX = re.compile(r"\bkontonummer\b", re.IGNORECASE)
KONTONUMMER_COLOR = "#FFEB3B"  # yellow highlight for the "Kontonummer" keyword
CONTEXT_WINDOW_SIZE = 300

# A pleasant, high-contrast palette. Cycled if there are more BAKs than colors.
COLOR_PALETTE = [
    "#FFD54F",  # amber
    "#4FC3F7",  # light blue
    "#81C784",  # green
    "#FF8A65",  # deep orange
    "#BA68C8",  # purple
    "#F06292",  # pink
    "#A1887F",  # brown
    "#4DB6AC",  # teal
    "#9575CD",  # deep purple
    "#DCE775",  # lime
]


def _safe_literal_eval(value, default):
    """Parse a stringified python literal, tolerating NaN / empty values."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return default
    if not isinstance(value, str):
        return value
    value = value.strip()
    if not value:
        return default
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return default


@st.cache_data
def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["bak_to_ibans"] = df["bak_to_ibans"].apply(lambda v: _safe_literal_eval(v, {}))
    df["bank_numbers"] = df["bank_numbers"].apply(lambda v: _safe_literal_eval(v, []))
    # df = df.iloc[[49,1,2]]
    # df.reset_index(drop=True, inplace=True)
    return df


def _ibans_as_list(value) -> list:
    """Normalize a BAK's mapped value into a list of IBAN strings.

    Supports the current shape (a single IBAN string or None) as well as the
    legacy shape (a list of IBAN strings).
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, (list, tuple, set)):
        return [v for v in value if v]
    return [value]


def _iter_iban_variants(iban: str):
    """Yield literal variants of an IBAN so we can find it in text regardless of spacing."""
    iban = str(iban)
    variants = {iban, iban.replace(" ", "")}
    # Also match the compact IBAN written with arbitrary single spaces between groups.
    compact = iban.replace(" ", "")
    if compact:
        spaced = re.escape(compact)
        # allow an optional space between every character group of 1+.
        spaced = r"\s*".join(re.escape(c) for c in compact)
        variants.add(("__REGEX__", spaced))
    return variants


def collect_spans(text: str, bak_to_ibans: dict) -> list:
    """Return a list of (start, end, color, kind) spans to highlight.

    A BAK and the IBANs mapped to it get the same color.
    """
    spans = []
    bak_color = {}
    for idx, bak in enumerate(bak_to_ibans.keys()):
        bak_color[str(bak)] = COLOR_PALETTE[idx % len(COLOR_PALETTE)]

    # BAK number spans (match the full "BAK-Nr: 123456" and color by its 6 digits).
    for match in DRITTAUSKUNFT_BAK_NUMBER_REGEX.finditer(text):
        digits_match = re.search(r"\d{6}", match.group())
        if not digits_match:
            continue
        bak = digits_match.group()
        if bak in bak_color:
            spans.append((match.start(), match.end(), bak_color[bak], "BAK"))

    # IBAN spans.
    for bak, ibans in bak_to_ibans.items():
        color = bak_color[str(bak)]
        for iban in _ibans_as_list(ibans):
            variants = _iter_iban_variants(iban)
            found = False
            # Try literal variants first.
            for variant in variants:
                if isinstance(variant, tuple):
                    continue
                start = 0
                while True:
                    pos = text.find(variant, start)
                    if pos == -1:
                        break
                    spans.append((pos, pos + len(variant), color, "IBAN"))
                    start = pos + len(variant)
                    found = True
            # Fall back to a spacing-tolerant regex if literal search failed.
            if not found:
                for variant in variants:
                    if isinstance(variant, tuple) and variant[0] == "__REGEX__":
                        for m in re.finditer(variant[1], text):
                            spans.append((m.start(), m.end(), color, "IBAN"))

    # Always highlight the "Kontonummer" keyword in yellow.
    for match in KONTONUMMER_REGEX.finditer(text):
        spans.append((match.start(), match.end(), KONTONUMMER_COLOR, "KONTONUMMER"))
    return spans


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Convert a #RRGGBB hex color to an rgba() string with the given alpha."""
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def compute_context_spans(text: str, bak_color: dict) -> list:
    """Replicate the extraction algorithm's context windows, one per BAK.

    Returns a list of (start, end, bak, color) covering the text region that the
    algorithm associates with each BAK number.
    """
    matches = list(DRITTAUSKUNFT_BAK_NUMBER_REGEX.finditer(text))

    # Deduplicate BAKs, keeping the first occurrence (same as `filter_bak_spans`).
    seen = set()
    filtered = []
    for match in matches:
        digits = re.search(r"\d{6}", match.group())
        if not digits:
            continue
        bak = digits.group()
        if bak not in seen:
            seen.add(bak)
            filtered.append((bak, match))

    spans = []
    for i, (bak, match) in enumerate(filtered):
        start = max(0, match.start())
        if i + 1 < len(filtered):
            end = min(len(text), filtered[i + 1][1].start())
        else:
            tail = text[start:]
            last_konto = deque(KONTOINHABER_REGEX.finditer(tail), maxlen=1)
            if last_konto:
                end = last_konto[0].start() + start
            else:
                end = start + CONTEXT_WINDOW_SIZE
        spans.append((start, min(end, len(text)), bak, bak_color.get(bak)))
    return spans


def _resolve_non_overlapping(spans: list) -> list:
    """Sort spans and drop any that overlap an already accepted span."""
    spans = sorted(spans, key=lambda s: (s[0], -(s[1] - s[0])))
    result = []
    last_end = -1
    for start, end, color, kind in spans:
        if start >= last_end:
            result.append((start, end, color, kind))
            last_end = end
    return result


def render_highlighted_text(text: str, bak_to_ibans: dict) -> str:
    """Build HTML with highlighted BAK / IBAN spans on top of shaded context windows.

    Each BAK's context window gets a light background tint; the BAK number and its
    IBANs are highlighted with the solid color on top.
    """
    if not text:
        return "<i>No text.</i>"

    bak_color = {
        str(bak): COLOR_PALETTE[idx % len(COLOR_PALETTE)]
        for idx, bak in enumerate(bak_to_ibans.keys())
    }

    token_spans = _resolve_non_overlapping(collect_spans(text, bak_to_ibans))
    context_spans = compute_context_spans(text, bak_color)

    n = len(text)
    ctx_bg = [None] * n  # per-char light context background (rgba string)
    for start, end, _bak, color in context_spans:
        if not color:
            continue
        tint = _hex_to_rgba(color, 0.18)
        for j in range(start, min(end, n)):
            ctx_bg[j] = tint

    token = [None] * n  # per-char (color, kind) for BAK / IBAN spans
    for start, end, color, kind in token_spans:
        for j in range(start, min(end, n)):
            token[j] = (color, kind)

    def _style_for(j):
        return (ctx_bg[j], token[j])

    pieces = []
    j = 0
    while j < n:
        cur = _style_for(j)
        k = j + 1
        while k < n and _style_for(k) == cur:
            k += 1
        chunk = html.escape(text[j:k])
        bg, tok = cur
        if tok is not None:
            color, kind = tok
            border = "2px solid rgba(0,0,0,0.35)" if kind == "IBAN" else "none"
            pieces.append(
                f'<span style="background-color:{color};border-radius:3px;'
                f'padding:0 2px;border:{border};font-weight:600;" '
                f'title="{kind}">{chunk}</span>'
            )
        elif bg is not None:
            pieces.append(f'<span style="background-color:{bg};">{chunk}</span>')
        else:
            pieces.append(chunk)
        j = k

    body = "".join(pieces).replace("\n", "<br>")
    return (
        '<div style="white-space:pre-wrap;font-family:monospace;font-size:13px;'
        'line-height:1.5;max-height:80vh;overflow-y:auto;padding:12px;'
        'border:1px solid #ddd;border-radius:6px;background:#fafafa;">'
        f"{body}</div>"
    )


def render_legend(bak_to_ibans: dict) -> str:
    items = []
    for idx, (bak, ibans) in enumerate(bak_to_ibans.items()):
        color = COLOR_PALETTE[idx % len(COLOR_PALETTE)]
        iban_list = _ibans_as_list(ibans)
        ibans_str = ", ".join(iban_list) if iban_list else "—"
        items.append(
            f'<div style="margin:2px 0;"><span style="display:inline-block;width:14px;'
            f'height:14px;background:{color};border-radius:3px;margin-right:6px;'
            f'vertical-align:middle;"></span>'
            f"<b>BAK {html.escape(str(bak))}</b> &rarr; {html.escape(ibans_str)}</div>"
        )
    if not items:
        return "<i>No BAK/IBAN mapping.</i>"
    return "".join(items)


def show_pdf(pdf_path: str):
    if not isinstance(pdf_path, str) or not pdf_path or not os.path.exists(pdf_path):
        st.warning(f"PDF not found: {pdf_path}")
        return
    with open(pdf_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode("utf-8")
    st.markdown(
        f'<iframe src="data:application/pdf;base64,{base64_pdf}" '
        'width="100%" height="800" style="border:1px solid #ddd;border-radius:6px;">'
        "</iframe>",
        unsafe_allow_html=True,
    )


def main():
    st.set_page_config(page_title="BAK → IBAN Inspector", layout="wide")
    st.title("Drittauskunft: BAK → IBAN extraction inspector")

    df = load_data(CSV_PATH)

    # Document selector.
    label_col = "ticket_uuid" if "ticket_uuid" in df.columns else None
    options = list(range(len(df)))

    def _fmt(i):
        if label_col:
            return f"{i} — {df.iloc[i][label_col]}"
        return str(i)

    selected = st.selectbox("Select document", options, format_func=_fmt)
    row = df.iloc[selected]

    bak_to_ibans = row["bak_to_ibans"] if isinstance(row["bak_to_ibans"], dict) else {}
    bank_numbers = row["bank_numbers"] if isinstance(row["bank_numbers"], list) else []

    # Top: column values.
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("no_result", str(row.get("no_result")))
    with c2:
        st.markdown("**bank_numbers**")
        st.write(bank_numbers if bank_numbers else "—")
    with c3:
        st.markdown("**bak_to_ibans**")
        st.write(bak_to_ibans if bak_to_ibans else "—")

    st.markdown("**Legend (color = BAK and its IBANs)**", unsafe_allow_html=True)
    st.markdown(render_legend(bak_to_ibans), unsafe_allow_html=True)

    st.divider()

    left, right = st.columns(2)
    with left:
        st.subheader("Document text")
        st.markdown(
            render_highlighted_text(str(row.get("text", "")), bak_to_ibans),
            unsafe_allow_html=True,
        )
    with right:
        st.subheader("PDF")
        show_pdf(row.get("local_file_path"))


if __name__ == "__main__":
    main()
