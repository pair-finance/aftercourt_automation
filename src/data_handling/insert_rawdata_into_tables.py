"""
Module: insert_rawdata_into_tables.py

Reads raw labeled data, deduplicates against existing DB records,
and inserts new rows into:
  - llm_label_defs   (document type labels)
  - llm_ticket_labels (per-attachment label assignments)
  - textract_jobs     (OCR job records)
"""

import hashlib
import logging
import uuid
from datetime import datetime
from typing import Optional

import pandas as pd
from python_utilities.db_connection import DbConnection
from sqlalchemy import text

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RAW_DATA_PATH = "data/raw/final_raw_data.csv"



# ---------------------------------------------------------------------------
# 1) Ensure label defs exist
# ---------------------------------------------------------------------------

def ensure_label_defs(db: DbConnection, document_types: list[str], dry_run: bool = False) -> dict[str, int]:
    """
    For each document_type, insert into llm_label_defs only if it doesn't
    already exist (type + subtype='none' + nature='aftercourt').

    Returns a mapping {document_type: label_id}.
    In dry_run mode, skips inserts and uses None as placeholder id for new labels.
    """
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    label_to_id: dict[str, int] = {}

    for doc_type in document_types:
        existing = db.sql_to_df(f"""
            SELECT id FROM llm_label_defs
            WHERE type = '{doc_type}'
              AND subtype = 'none'
              AND nature = 'aftercourt'
            LIMIT 1
        """)

        if len(existing) > 0:
            label_id = int(existing.iloc[0]["id"])
            logger.info(f"Label '{doc_type}' already exists with id={label_id}")
        else:
            if dry_run:
                logger.info(f"[DRY RUN] Would insert label '{doc_type}' into llm_label_defs")
                label_to_id[doc_type] = None
                continue

            insert_q = text("""
                INSERT INTO llm_label_defs (type, subtype, nature, created_at, updated_at, datatype)
                VALUES (:type, :subtype, :nature, :created_at, :updated_at, NULL)
            """)
            with db.engine.connect() as conn:
                conn.execute(insert_q, {
                    "type": doc_type,
                    "subtype": "none",
                    "nature": "aftercourt",
                    "created_at": now,
                    "updated_at": now,
                })
                conn.commit()

            # fetch the auto-generated id
            result = db.sql_to_df(f"""
                SELECT id FROM llm_label_defs
                WHERE type = '{doc_type}'
                  AND subtype = 'none'
                  AND nature = 'aftercourt'
                LIMIT 1
            """)
            label_id = int(result.iloc[0]["id"])
            logger.info(f"Inserted label '{doc_type}' → id={label_id}")

        label_to_id[doc_type] = label_id

    return label_to_id


# ---------------------------------------------------------------------------
# 2) Filter out rows already in llm_ticket_labels
# ---------------------------------------------------------------------------

def filter_existing_ticket_labels(db: DbConnection, df: pd.DataFrame) -> pd.DataFrame:
    """
    Query llm_ticket_labels for the ticket_uuids in *df*.
    Remove any rows whose ticket_uuid already has a label entry.
    """
    ticket_uuids = df["ticket_uuid"].unique().tolist()
    if not ticket_uuids:
        return df

    # batch in chunks of 500 to avoid huge IN clauses
    existing_uuids: set[str] = set()
    chunk_size = 500
    for i in range(0, len(ticket_uuids), chunk_size):
        chunk = ticket_uuids[i : i + chunk_size]
        in_clause = ",".join(f"'{uid}'" for uid in chunk)
        result = db.sql_to_df(f"""
            SELECT DISTINCT ticket_uuid
            FROM llm_ticket_labels
            WHERE ticket_uuid IN ({in_clause})
        """)
        existing_uuids.update(result["ticket_uuid"].tolist())

    before = len(df)
    df_new = df[~df["ticket_uuid"].isin(existing_uuids)].copy()
    logger.info(
        f"Filtered ticket_labels: {before} total, {before - len(df_new)} already exist, "
        f"{len(df_new)} new rows to insert"
    )
    return df_new


# ---------------------------------------------------------------------------
# 3) Insert into llm_ticket_labels
# ---------------------------------------------------------------------------

def insert_ticket_labels(db: DbConnection, df: pd.DataFrame, label_to_id: dict[str, int]) -> int:
    """Insert new rows into llm_ticket_labels. Returns count of inserted rows."""
    if df.empty:
        logger.info("No new ticket labels to insert.")
        return 0

    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    insert_q = text("""
        INSERT INTO llm_ticket_labels
            (ticket_uuid, zendesk_id, comment_id, attachment_id, label_id, value, created_at, updated_at)
        VALUES
            (:ticket_uuid, NULL, NULL, :attachment_id, :label_id, :value, :created_at, :updated_at)
    """)

    inserted = 0
    with db.engine.connect() as conn:
        for _, row in df.iterrows():
            label_id = label_to_id.get(row["document_type"])
            if label_id is None:
                logger.warning(f"No label_id for document_type='{row['document_type']}', skipping")
                continue
            conn.execute(insert_q, {
                "ticket_uuid": str(row["ticket_uuid"]),
                "attachment_id": str(row["attachment_id"]),
                "label_id": label_id,
                "value": "True",
                "created_at": now,
                "updated_at": now,
            })
            inserted += 1
        conn.commit()

    logger.info(f"Inserted {inserted} rows into llm_ticket_labels")
    return inserted


# ---------------------------------------------------------------------------
# 4) Insert into textract_jobs
# ---------------------------------------------------------------------------

def insert_textract_jobs(db: DbConnection, df: pd.DataFrame) -> int:
    """Insert new rows into textract_jobs. Returns count of inserted rows."""
    if df.empty:
        logger.info("No new textract jobs to insert.")
        return 0

    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    insert_q = text("""
        INSERT INTO textract_jobs
            (attachment_id, job_id, status, s3_link, created_at, updated_at)
        VALUES
            (:attachment_id, :job_id, :status, :s3_link, :created_at, :updated_at)
    """)

    inserted = 0
    with db.engine.connect() as conn:
        for _, row in df.iterrows():
            conn.execute(insert_q, {
                "attachment_id": str(row["attachment_id"]),
                "job_id": str(row["textract_job_id"]),
                "status": "SUCCEEDED",
                "s3_link": str(row["textract_s3_link"]),
                "created_at": now,
                "updated_at": now,
            })
            inserted += 1
        conn.commit()

    logger.info(f"Inserted {inserted} rows into textract_jobs")
    return inserted


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run(
    raw_data_path: str = RAW_DATA_PATH,
    db_section: str = "ANALYTICS",
    db_option: str = "PROD_RDS",
    dry_run: bool = False,
):
    """
    End-to-end pipeline:
      2. Ensure all document-type labels exist in llm_label_defs
      3. Filter out tickets already present in llm_ticket_labels
      4. Insert new ticket labels
      5. Insert new textract jobs
    """
    db = DbConnection(db_section, db_option)

    # Step 1 – read raw data and generate missing idss
    df = pd.read_csv(raw_data_path)
    
    # data to be inserted should have ticket_uuid, attachment_id, textract_job_id, textract_s3_link
    
    assert df['ticket_uuid'].isna().sum() == 0, "There is ticket uuid with none value, check the input data"
    assert df['attachment_id'].isna().sum() == 0, "There is attachment_id with none value, check the input data"
    assert df['textract_job_id'].isna().sum() == 0, "There is textract_job_id with none value, check the input data"
    assert df['textract_s3_link'].isna().sum() == 0, "There is textract_s3_link with none value, check the input data"

    # remove mail_attachments document type
    df = df[df['document_type'] != 'mail_attachments']

    # Step 2 – ensure label defs exist (inserts only missing ones)
    document_types = df["document_type"].unique().tolist()
    label_to_id = ensure_label_defs(db, document_types, dry_run=dry_run)
    logger.info(f"Label mapping: {label_to_id}")

    # Step 3 – filter out already-labeled tickets
    df_new = filter_existing_ticket_labels(db, df)

    if df_new.empty:
        logger.info("Nothing new to insert. Done.")
        return

    if dry_run:
        logger.info(f"[DRY RUN] Would insert {len(df_new)} rows into llm_ticket_labels and textract_jobs")
        print(df_new[["ticket_uuid", "attachment_id", "document_type", "textract_job_id"]].to_string(index=False))
        return

    # Step 4 – insert into llm_ticket_labels
    insert_ticket_labels(db, df_new, label_to_id)

    # Step 5 – insert into textract_jobs
    insert_textract_jobs(db, df_new)

    logger.info("Pipeline complete.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Insert raw labeled data into prod tables")
    parser.add_argument("--raw-data-path", default=RAW_DATA_PATH, help="Path to final_raw_data.csv")
    parser.add_argument("--db-section", default="ANALYTICS", help="DbConnection section")
    parser.add_argument("--db-option", default="PROD_RDS", help="DbConnection option")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be inserted without writing")
    args = parser.parse_args()

    run(
        raw_data_path=args.raw_data_path,
        db_section=args.db_section,
        db_option=args.db_option,
        dry_run=args.dry_run,
    )
