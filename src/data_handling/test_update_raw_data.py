import sys
import os
import pytest
import numpy as np
import pandas as pd

# Ensure project root is on sys.path so imports resolve
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.data_handling.update_raw_data import (
    validate_required_columns,
    fill_generated_columns,
    deduplicate_against_existing,
    update_raw_data,
    REQUIRED_COLUMNS,
)
from utils.data_handling_utils import generate_uuid, generate_hash_from_text


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_new_data(**overrides) -> pd.DataFrame:
    """Return a minimal valid new-data DataFrame."""
    base = {
        "text": ["Hello world", "Foo bar baz"],
        "document_type": ["ladung", "pfub"],
        "is_pfub": [0, 1],
        "is_ladung": [1, 0],
    }
    base.update(overrides)
    return pd.DataFrame(base)


def _make_raw_data(n: int = 1) -> pd.DataFrame:
    """Return a small raw-data DataFrame with the expected columns."""
    return pd.DataFrame({
        "text": [f"existing text {i}" for i in range(n)],
        "document_type": ["ladung"] * n,
        "is_pfub": [0] * n,
        "is_ladung": [1] * n,
        "ticket_uuid": [generate_uuid(f"existing text {i}", "ticket_uuid") for i in range(n)],
        "attachment_id": [generate_uuid(f"existing text {i}", "attachment_id") for i in range(n)],
        "textract_job_id": [generate_hash_from_text(f"existing text {i}") for i in range(n)],
        "textract_s3_link": [
            f"s3://pair-data-engineering-new/ocr_prepared_output/{generate_hash_from_text(f'existing text {i}')}.json"
            for i in range(n)
        ],
    })


# ── validate_required_columns ────────────────────────────────────────────────

class TestValidateRequiredColumns:
    def test_passes_when_all_present(self):
        df = _make_new_data()
        validate_required_columns(df)  # should not raise

    @pytest.mark.parametrize("col", REQUIRED_COLUMNS)
    def test_fails_when_column_has_nan(self, col):
        df = _make_new_data()
        df.loc[0, col] = np.nan
        with pytest.raises(AssertionError, match="empty entry"):
            validate_required_columns(df)


# ── fill_generated_columns ───────────────────────────────────────────────────

class TestFillGeneratedColumns:
    def test_creates_all_columns_when_absent(self):
        df = _make_new_data()
        result = fill_generated_columns(df)

        for col in ["ticket_uuid", "attachment_id", "textract_job_id", "textract_s3_link"]:
            assert col in result.columns, f"Missing column: {col}"
            assert result[col].isna().sum() == 0, f"NaN found in {col}"

    def test_ticket_uuid_is_deterministic(self):
        df = _make_new_data()
        r1 = fill_generated_columns(df.copy())
        r2 = fill_generated_columns(df.copy())
        assert list(r1["ticket_uuid"]) == list(r2["ticket_uuid"])

    def test_preserves_existing_non_nan_values(self):
        df = _make_new_data(ticket_uuid=["my-existing-uuid", np.nan])
        result = fill_generated_columns(df)

        assert result.loc[0, "ticket_uuid"] == "my-existing-uuid"
        assert pd.notna(result.loc[1, "ticket_uuid"])
        assert result.loc[1, "ticket_uuid"] != "my-existing-uuid"

    def test_textract_job_id_is_sha256_hash(self):
        df = _make_new_data()
        result = fill_generated_columns(df)

        for job_id in result["textract_job_id"]:
            assert len(job_id) == 64, "SHA-256 hash should be 64 hex chars"
            assert all(c in "0123456789abcdef" for c in job_id)

    def test_textract_s3_link_format(self):
        df = _make_new_data()
        result = fill_generated_columns(df)

        for _, row in result.iterrows():
            expected = f"s3://pair-data-engineering-new/ocr_prepared_output/{row['textract_job_id']}.json"
            assert row["textract_s3_link"] == expected

    def test_preserves_existing_s3_link(self):
        custom_link = "s3://custom-bucket/custom-path.json"
        df = _make_new_data(textract_s3_link=[custom_link, np.nan])
        # need textract_job_id to exist for s3 link derivation
        result = fill_generated_columns(df)

        assert result.loc[0, "textract_s3_link"] == custom_link
        assert result.loc[1, "textract_s3_link"].startswith("s3://pair-data-engineering-new/")


# ── deduplicate_against_existing ─────────────────────────────────────────────

class TestDeduplicateAgainstExisting:
    def test_removes_duplicates(self):
        raw = _make_raw_data(2)
        new = _make_new_data()
        new = fill_generated_columns(new)
        # inject a duplicate ticket_uuid
        new.loc[0, "ticket_uuid"] = raw.loc[0, "ticket_uuid"]

        result = deduplicate_against_existing(raw, new)
        assert len(result) == 1
        assert raw.loc[0, "ticket_uuid"] not in set(result["ticket_uuid"])

    def test_keeps_all_when_no_duplicates(self):
        raw = _make_raw_data(1)
        new = fill_generated_columns(_make_new_data())

        result = deduplicate_against_existing(raw, new)
        assert len(result) == len(new)


# ── update_raw_data (integration) ───────────────────────────────────────────

class TestUpdateRawData:
    def test_full_pipeline_adds_new_rows(self):
        raw = _make_raw_data(1)
        new = _make_new_data()

        result = update_raw_data(raw, new)
        assert len(result) == len(raw) + len(new)

    def test_full_pipeline_no_duplicates(self):
        raw = _make_raw_data(1)
        # add data whose text matches existing → same ticket_uuid after generation
        new = _make_new_data(text=["existing text 0", "brand new text"])
        result = update_raw_data(raw, new)

        # "existing text 0" should be deduplicated, "brand new text" kept
        assert len(result) == len(raw) + 1

    def test_raises_on_invalid_new_data(self):
        raw = _make_raw_data(1)
        bad = _make_new_data()
        bad.loc[0, "text"] = np.nan

        with pytest.raises(AssertionError):
            update_raw_data(raw, bad)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
