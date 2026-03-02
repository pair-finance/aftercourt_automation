import pandas as pd
from utils.data_handling_utils import *

RAW_DATA_PATH = "/Users/melih.gorgulu/Desktop/Projects/aftercourt_automation/data/raw/final_raw_data.csv"

REQUIRED_COLUMNS = ['text', 'document_type', 'is_pfub', 'is_ladung']


def validate_required_columns(data: pd.DataFrame) -> None:
    """Validate that required columns have no missing values."""
    missing = data[REQUIRED_COLUMNS].isna().sum()
    assert missing.sum() == 0, f"There is empty entry for required fields in new data. Check fields: {REQUIRED_COLUMNS}"


def _fill_or_create_uuid_column(data: pd.DataFrame, column: str, purpose: str) -> pd.DataFrame:
    """Generate UUIDs for a column, preserving existing non-NaN values."""
    if column in data.columns:
        data[column] = data.apply(
            lambda row: generate_uuid(row['text'], purpose=purpose) if pd.isna(row[column]) else row[column], axis=1
        )
    else:
        data[column] = data.apply(lambda row: generate_uuid(row['text'], purpose=purpose), axis=1)
    return data


def fill_generated_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Fill ticket_uuid, attachment_id, textract_job_id, and textract_s3_link."""
    data = _fill_or_create_uuid_column(data, 'ticket_uuid', 'ticket_uuid')
    data = _fill_or_create_uuid_column(data, 'attachment_id', 'attachment_id')

    # textract_job_id uses a hash instead of uuid
    if "textract_job_id" in data.columns:
        data['textract_job_id'] = data.apply(
            lambda row: generate_hash_from_text(row['text']) if pd.isna(row['textract_job_id']) else row['textract_job_id'], axis=1
        )
    else:
        data['textract_job_id'] = data.apply(lambda row: generate_hash_from_text(row['text']), axis=1)

    # textract_s3_link is derived from textract_job_id
    if "textract_s3_link" in data.columns:
        data['textract_s3_link'] = data.apply(
            lambda row: f"s3://pair-data-engineering-new/ocr_prepared_output/{row['textract_job_id']}.json"
            if pd.isna(row['textract_s3_link']) else row['textract_s3_link'], axis=1
        )
    else:
        data['textract_s3_link'] = data['textract_job_id'].apply(
            lambda x: f"s3://pair-data-engineering-new/ocr_prepared_output/{x}.json"
        )

    return data


def validate_dtypes(raw_data: pd.DataFrame, new_data: pd.DataFrame) -> None:
    """Assert that columns shared between new_data and raw_data have matching dtypes."""
    shared_cols = [col for col in new_data.columns if col in raw_data.columns]
    mismatches = {
        col: {"raw_data": str(raw_data[col].dtype), "new_data": str(new_data[col].dtype)}
        for col in shared_cols
        if raw_data[col].dtype != new_data[col].dtype
    }
    assert len(mismatches) == 0, (
        f"Dtype mismatch between raw data and new data for columns: {mismatches}. "
        "Fix the new data dtypes before inserting."
    )


def deduplicate_against_existing(raw_data: pd.DataFrame, new_data: pd.DataFrame) -> pd.DataFrame:
    """Remove rows from new_data whose ticket_uuid already exists in raw_data."""
    existing_uuids = set(raw_data['ticket_uuid'])
    return new_data[~new_data['ticket_uuid'].isin(existing_uuids)]


def update_raw_data(raw_data: pd.DataFrame, new_data: pd.DataFrame) -> pd.DataFrame:
    """Full pipeline: validate, fill columns, deduplicate, and concatenate."""
    validate_required_columns(new_data)
    new_data = fill_generated_columns(new_data)
    validate_dtypes(raw_data, new_data)
    new_data = deduplicate_against_existing(raw_data, new_data)
    return pd.concat([raw_data, new_data], ignore_index=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser("Update raw data with new data")
    parser.add_argument("--raw-data-path", default=RAW_DATA_PATH, help="Path to final_raw_data.csv")
    parser.add_argument("--new-data-path", help="Path to data to be added")
    args = parser.parse_args()

    raw_data = pd.read_csv(args.raw_data_path)
    data_to_add = pd.read_csv(args.new_data_path)

    updated_raw_data = update_raw_data(raw_data, data_to_add)
    #updated_raw_data.to_csv(args.raw_data_path, index=False)
    print(f"Updated raw data saved to {args.raw_data_path}. Added {len(updated_raw_data) - len(raw_data)} new rows.")
    