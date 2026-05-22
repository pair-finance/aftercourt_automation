"""Heuristic correction of LLM invoice page-range predictions."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Mapping, Optional, Tuple

import pandas as pd

from intent_recognition.src.services.attachment_processing.aftercourt_extractors.ladung.slug_extractor import (
    SlugExtractor,
)

_SLUG_EXTRACTOR = SlugExtractor()


# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------

# German page markers: "Seite 1", "Seite: 1", "Seite 1/3", "Seite 1 von 3", "Blatt 1"
_SEITE_PATTERNS = [
    r'(?i)\bseite\s*:?\s*\d+',         # Seite 1, Seite: 1
    r'(?i)\bseite\s*\d+\s*/\s*\d+',    # Seite 1/3
    r'(?i)\bseite\s*\d+\s*von\s*\d+',  # Seite 1 von 3
    r'(?i)\bblatt\s*:?\s*\d+',         # Blatt 1
]
_SEITE_REGEXES = [re.compile(p) for p in _SEITE_PATTERNS]

_PAGE_TAG_RE = re.compile(r'<page_(\d+)>')
_DIGIT_RE = re.compile(r'\d+')

# Guard pattern: keywords whose presence on the *next* page indicates that
# the page belongs to a different document (e.g. asset disclosure / protocol),
# so we should NOT extend the invoice range to include it.
GUARD_PATTERN = (
    r"(?im)^\s*(?:protokoll"
    r"|verm[oö0]gens\s*verzeichnis"
    r"|verm[oö0]gens\s*auskunft(?:s\s*protokoll|protokoll)?"
    r"|dritt\s*ausk[uü]nfte"
    r"|ergebnis(?:se)?(?:\s+der\s+verm[oö0]gens\s*auskunft)?"
    r")\b"
)
GUARD_RE = re.compile(GUARD_PATTERN)

# Default thresholds
DEFAULT_SHORT_PAGE_WORD_THRESHOLD = 100
NO_CHANGE_SENTINEL = -1


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def check_seite_marker_inside(text: str) -> Tuple[bool, Optional[str]]:
    """Return ``(found, matched_string)`` for the first German page marker."""
    for regex in _SEITE_REGEXES:
        match = regex.search(text)
        if match:
            return True, match.group()
    return False, None


def extract_page_text(full_text: str, page_idx: int, number_of_pages: int) -> str:
    """Extract the text of a single page delimited by ``<page_N>`` markers."""
    if page_idx == number_of_pages:
        start_marker = f"<page_{page_idx}>"
        if start_marker in full_text:
            return full_text.split(start_marker)[-1]
        return ""

    start_marker = f"<page_{page_idx}>"
    end_marker = f"<page_{page_idx + 1}>"
    if start_marker in full_text and end_marker in full_text:
        return full_text.split(start_marker)[-1].split(end_marker)[0]
    return ""


def _count_pages(clean_text: str) -> int:
    """Total number of pages in the document, inferred from ``<page_N>`` tags."""
    nums = [int(n) for n in _PAGE_TAG_RE.findall(clean_text)]
    return max(nums) if nums else 0


def _arg_find_consecutive_sequence_for_invoice(found_page_markers):
    """Return indices of the longest consecutive marker sequence starting at 0."""
    if not found_page_markers:
        return []

    first_number_match = _DIGIT_RE.search(found_page_markers[0])
    if not first_number_match:
        return []
    first_number = int(first_number_match.group())
    first_pattern = _DIGIT_RE.sub('', found_page_markers[0]).strip().lower()

    for i in range(1, len(found_page_markers)):
        current_marker = found_page_markers[i]
        current_pattern = _DIGIT_RE.sub('', current_marker).strip().lower()
        if current_pattern != first_pattern:
            return list(range(i))
        current_number_match = _DIGIT_RE.search(current_marker)
        if not current_number_match:
            return list(range(i))
        if int(current_number_match.group()) != first_number + i:
            return list(range(i))

    return list(range(len(found_page_markers)))


# ---------------------------------------------------------------------------
# Row access (DataFrame row OR plain dict)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _RowFields:
    pred_is_invoice_inside: bool
    pred_start_page: int
    pred_end_page: int
    clean_text: str


def _read_row(row: Mapping) -> _RowFields:
    return _RowFields(
        pred_is_invoice_inside=bool(row['pred_is_invoice_inside']),
        pred_start_page=int(row['pred_start_page']),
        pred_end_page=int(row['pred_end_page']),
        clean_text=row['clean_text'] or "",
    )


# ---------------------------------------------------------------------------
# Heuristic 1: page-marker following
# ---------------------------------------------------------------------------

def include_if_page_number_exists(row: Mapping) -> Tuple[int, int]:
    """Extend the invoice range using consecutive German page markers.

    Returns ``(start_page_idx, end_page_idx)``. Returns
    ``(NO_CHANGE_SENTINEL, NO_CHANGE_SENTINEL)`` when the row is not predicted
    to contain an invoice.
    """
    fields = _read_row(row)
    if not fields.pred_is_invoice_inside:
        return NO_CHANGE_SENTINEL, NO_CHANGE_SENTINEL

    n_of_pages = _count_pages(fields.clean_text)
    start_page_idx = fields.pred_start_page
    end_page_idx = fields.pred_end_page

    invoice_page_text = extract_page_text(
        fields.clean_text, start_page_idx, n_of_pages
    )

    # ----- Forward iteration -----
    cur_text = invoice_page_text
    cur_page_idx = start_page_idx
    found_page_markers = []
    found_page_markers_page_indices = []

    while True:
        is_with_marker, matched = check_seite_marker_inside(cur_text)
        if not (is_with_marker and matched):
            break
        found_page_markers.append(matched)
        found_page_markers_page_indices.append(cur_page_idx)
        cur_page_idx += 1
        cur_text = extract_page_text(fields.clean_text, cur_page_idx, n_of_pages)

    indexes = _arg_find_consecutive_sequence_for_invoice(found_page_markers)
    if indexes:
        end_page_idx = found_page_markers_page_indices[indexes[-1]]

    # ----- Backward iteration -----
    is_with_marker, matched_start_marker = check_seite_marker_inside(invoice_page_text)
    matched_number = (
        _DIGIT_RE.search(matched_start_marker) if matched_start_marker else None
    )

    if (
        is_with_marker
        and matched_start_marker
        and matched_number
        and int(matched_number.group()) > 1
    ):
        start_pattern = _DIGIT_RE.sub('', matched_start_marker).strip().lower()
        expected_number = int(matched_number.group()) - 1
        prev_page_idx = start_page_idx - 1

        while prev_page_idx >= 1 and expected_number > 0:
            prev_page_text = extract_page_text(
                fields.clean_text, prev_page_idx, n_of_pages
            )
            is_with_prev_marker, matched_prev = check_seite_marker_inside(prev_page_text)
            if not (is_with_prev_marker and matched_prev):
                break

            prev_pattern = _DIGIT_RE.sub('', matched_prev).strip().lower()
            if prev_pattern != start_pattern:
                break

            prev_number_match = _DIGIT_RE.search(matched_prev)
            if not prev_number_match:
                break
            if int(prev_number_match.group()) != expected_number:
                break

            start_page_idx = prev_page_idx
            expected_number -= 1
            prev_page_idx -= 1

    return start_page_idx, end_page_idx


# ---------------------------------------------------------------------------
# Heuristic 2: short next page
# ---------------------------------------------------------------------------

def include_if_next_page_short(
    row: Mapping,
    word_threshold: int = DEFAULT_SHORT_PAGE_WORD_THRESHOLD,
) -> int:
    """Extend the end page by one if the following page is short.

    Skipped (returns the original end page) when the next page contains a
    guard keyword (see :data:`GUARD_RE`). Returns
    :data:`NO_CHANGE_SENTINEL` for rows not predicted as invoices.
    """
    fields = _read_row(row)
    if not fields.pred_is_invoice_inside:
        return NO_CHANGE_SENTINEL

    start_page_idx = fields.pred_start_page
    end_page_idx = fields.pred_end_page
    if start_page_idx != end_page_idx:
        return end_page_idx

    next_page_idx = start_page_idx + 1
    n_of_pages = _count_pages(fields.clean_text)
    next_page_text = extract_page_text(
        fields.clean_text, next_page_idx, n_of_pages
    )

    if not next_page_text:
        return end_page_idx

    if len(next_page_text.split()) >= word_threshold:
        return end_page_idx

    if GUARD_RE.search(next_page_text):
        return end_page_idx

    return next_page_idx


# ---------------------------------------------------------------------------
# Heuristic 3: slug-based margin extension
# ---------------------------------------------------------------------------

def _is_slug_found_in_range(
    clean_text: str, start_page_idx: int, end_page_idx: int, n_of_pages: int,
) -> bool:
    """Return True if a slug is extracted on any page in ``[start, end]``."""
    for page_idx in range(start_page_idx, end_page_idx + 1):
        page_text = extract_page_text(clean_text, page_idx, n_of_pages)
        if _SLUG_EXTRACTOR.extract(page_text):
            return True
    return False


def _extend_by_slug_in_margin(
    start_page_idx: int, end_page_idx: int, clean_text: str,
) -> Tuple[int, int]:
    """Extend range by one page on either side if the slug appears there.

    If no slug is found anywhere in the current ``[start, end]`` range, check
    the page immediately before ``start`` and the page immediately after
    ``end``; extend in whichever direction a slug is found.
    """
    if start_page_idx < 1 or end_page_idx < 1:
        return start_page_idx, end_page_idx

    n_of_pages = _count_pages(clean_text)
    if _is_slug_found_in_range(
        clean_text, start_page_idx, end_page_idx, n_of_pages
    ):
        return start_page_idx, end_page_idx

    check_start = max(1, start_page_idx - 1)
    if check_start != start_page_idx and _is_slug_found_in_range(
        clean_text, check_start, check_start, n_of_pages
    ):
        return check_start, end_page_idx

    check_end = min(end_page_idx + 1, n_of_pages) if n_of_pages else end_page_idx
    if check_end != end_page_idx and _is_slug_found_in_range(
        clean_text, check_end, check_end, n_of_pages
    ):
        return start_page_idx, check_end

    return start_page_idx, end_page_idx


def include_by_slug_in_margin(row: Mapping) -> Tuple[int, int]:
    """Row-level wrapper for :func:`_extend_by_slug_in_margin`.

    Returns ``(NO_CHANGE_SENTINEL, NO_CHANGE_SENTINEL)`` when the row is not
    predicted to contain an invoice.
    """
    fields = _read_row(row)
    if not fields.pred_is_invoice_inside:
        return NO_CHANGE_SENTINEL, NO_CHANGE_SENTINEL
    return _extend_by_slug_in_margin(
        fields.pred_start_page, fields.pred_end_page, fields.clean_text,
    )


# ---------------------------------------------------------------------------
# Combined entry point
# ---------------------------------------------------------------------------

def apply_correction(
    row: Mapping,
    short_page_word_threshold: int = DEFAULT_SHORT_PAGE_WORD_THRESHOLD,
) -> Tuple[int, int]:
    """Apply all heuristics in sequence and return ``(start, end)`` pages.

    The page-marker heuristic runs first. If it does not change the end page,
    the short-next-page heuristic is applied as a fallback. Finally, the
    slug-margin heuristic may extend the range by one page on either side.
    Returns ``(NO_CHANGE_SENTINEL, NO_CHANGE_SENTINEL)`` for rows not
    predicted to contain an invoice.
    """
    fields = _read_row(row)
    if not fields.pred_is_invoice_inside:
        return NO_CHANGE_SENTINEL, NO_CHANGE_SENTINEL

    start_page_idx, end_page_idx = include_if_page_number_exists(row)
    if end_page_idx == fields.pred_end_page:
        end_page_idx = include_if_next_page_short(
            row, word_threshold=short_page_word_threshold
        )

    start_page_idx, end_page_idx = _extend_by_slug_in_margin(
        start_page_idx, end_page_idx, fields.clean_text,
    )
    return start_page_idx, end_page_idx


def apply_correction_to_dataframe(
    df: pd.DataFrame,
    start_col: str = "corrected_start_page",
    end_col: str = "corrected_end_page",
    short_page_word_threshold: int = DEFAULT_SHORT_PAGE_WORD_THRESHOLD,
) -> pd.DataFrame:
    """Apply :func:`apply_correction` row-wise and add corrected columns.

    Expects the DataFrame to contain ``pred_is_invoice_inside``,
    ``pred_start_page``, ``pred_end_page`` and ``clean_text`` columns.
    Returns the same DataFrame for chaining.
    """
    corrected = df.apply(
        lambda row: apply_correction(
            row, short_page_word_threshold=short_page_word_threshold
        ),
        axis=1,
    )
    df[start_col] = corrected.apply(lambda t: t[0])
    df[end_col] = corrected.apply(lambda t: t[1])
    return df
