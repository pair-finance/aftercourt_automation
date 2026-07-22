"""Mark invoice pages for zendesk (attachment_id containing "-") PDFs.

For every attachment whose ``attachment_id`` contains a ``-`` it:
    1. looks up the invoice start/end page from ``invoice_detection_egvp``
    2. draws a green border around every invoice page (start..end inclusive)
    3. saves the PDF (real vector pages, not rasterized) back under
       ``only_zendesk`` using the same ``<attachment_id>.pdf`` name.

Output dir: ``assets/pdfs/tmp/unknown_rejected/only_zendesk/``

Run with:
    python notebooks/classification/invoice_page_detection/merge_zendesk_invoice_pdfs.py
"""
from __future__ import annotations

from pathlib import Path

import fitz  # PyMuPDF
import pandas as pd

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PDF_DIR = Path(
    "/Users/melih.gorgulu/Desktop/Projects/aftercourt_automation/assets/pdfs/tmp/unknown_rejected"
)
PREDICTIONS_CSV = PDF_DIR / "unknown_rejected_all_predictions.csv"
OUT_DIR = PDF_DIR / "only_zendesk"

GREEN = (0, 0.7, 0)
BORDER_INSET = 4.0
BORDER_WIDTH = 8.0


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _pred_value(att_preds: pd.DataFrame, model_name: str, subtype: str) -> str:
    vals = att_preds[
        (att_preds["model_name"] == model_name) & (att_preds["subtype"] == subtype)
    ]["value_clean"].values
    return vals[0] if len(vals) else "N/A"


def _to_int(value: str) -> int | None:
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return None


def _invoice_page_range(att_preds: pd.DataFrame) -> tuple[int, ...]:
    """1-based page numbers start..end (inclusive) for the invoice, or empty."""
    start = _to_int(_pred_value(att_preds, "invoice_detection_egvp", "start_page"))
    end = _to_int(_pred_value(att_preds, "invoice_detection_egvp", "end_page"))
    if start is None or end is None or start < 1 or end < start:
        return ()
    return tuple(range(start, end + 1))


def _resolve_pdf_path(att_id: str, att_preds: pd.DataFrame) -> Path:
    pdf_path = PDF_DIR / f"{att_id}.pdf"
    if not att_preds.empty and "pdf_path" in att_preds.columns:
        stored = str(att_preds["pdf_path"].iloc[0])
        if stored and stored != "nan" and Path(stored).exists():
            pdf_path = Path(stored)
    return pdf_path


def _draw_invoice_borders(doc: fitz.Document, invoice_pages: tuple[int, ...]) -> None:
    invoice_set = set(invoice_pages)
    for page in doc:
        if (page.number + 1) not in invoice_set:
            continue
        border = fitz.Rect(
            page.rect.x0 + BORDER_INSET,
            page.rect.y0 + BORDER_INSET,
            page.rect.x1 - BORDER_INSET,
            page.rect.y1 - BORDER_INSET,
        )
        page.draw_rect(border, color=GREEN, width=BORDER_WIDTH)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    df = pd.read_csv(PREDICTIONS_CSV)
    df["value_clean"] = df["value"].astype(str).str.strip("'")
    df["attachment_id"] = df["attachment_id"].astype(str)

    # Only zendesk attachments (attachment_id contains "-").
    zendesk_ids = sorted(a for a in df["attachment_id"].unique() if "-" in a)
    print(f"Found {len(zendesk_ids)} zendesk attachment(s) with '-' in the id.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    saved_count = 0
    skipped: list[str] = []
    for att_id in zendesk_ids:
        att_preds = df[df["attachment_id"] == att_id]
        pdf_path = _resolve_pdf_path(att_id, att_preds)
        if not pdf_path.exists():
            skipped.append(f"{att_id} (pdf not found: {pdf_path})")
            continue

        invoice_pages = _invoice_page_range(att_preds)
        doc = fitz.open(pdf_path)
        try:
            _draw_invoice_borders(doc, invoice_pages)
            out_path = OUT_DIR / f"{att_id}.pdf"
            doc.save(out_path, deflate=True, garbage=4)
        finally:
            doc.close()

        rng = f"{invoice_pages[0]}-{invoice_pages[-1]}" if invoice_pages else "none"
        print(f"  + {att_id}: invoice pages {rng} -> {out_path.name}")
        saved_count += 1

    print(f"\nSaved {saved_count} document(s) -> {OUT_DIR}")
    if skipped:
        print(f"Skipped {len(skipped)}:")
        for s in skipped:
            print(f"  - {s}")


if __name__ == "__main__":
    main()
