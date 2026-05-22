"""Invoice detection correction heuristics.

Post-processing utilities that refine LLM invoice page-range predictions
using two heuristics:

1. ``include_if_page_number_exists`` – Extends the predicted invoice range
   forward and backward by following consecutive German page markers
   (e.g. ``Seite 1``, ``Seite 1/3``, ``Seite 1 von 3``, ``Blatt 1``).
2. ``include_if_next_page_short`` – Extends the predicted end page by one
   when the next page is very short (likely a continuation/footer page),
   unless the next page contains guard keywords that indicate a different
   document type (e.g. ``Protokoll``, ``Vermögensverzeichnis``).

The combined entry point is :func:`apply_correction`, which can be applied
row-wise to a DataFrame produced by the invoice detection LLM pipeline.
"""

from .correction import (
    apply_correction,
    apply_correction_to_dataframe,
    check_seite_marker_inside,
    extract_page_text,
    include_by_slug_in_margin,
    include_if_next_page_short,
    include_if_page_number_exists,
)

__all__ = [
    "apply_correction",
    "apply_correction_to_dataframe",
    "check_seite_marker_inside",
    "extract_page_text",
    "include_by_slug_in_margin",
    "include_if_next_page_short",
    "include_if_page_number_exists",
]
