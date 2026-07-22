"""Download the PDFs listed in every ``*.csv`` under the ``csv`` folder.

For every ``attachment_id`` found across the dataset CSVs the corresponding PDF
is fetched from S3 using :func:`utils.prod_utils.get_data_by_attachment_id` and
stored in the ``pdf`` folder as ``<attachment_id>.pdf``.

Run with:
    python download_pdfs.py
"""
import glob
import os
import sys

import boto3
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CODE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CODE_DIR)  # tmp_streamlit_show
CSV_DIR = os.path.join(BASE_DIR, "csv")
DOWNLOAD_DIR = os.path.join(BASE_DIR, "pdf")

# Make the repo root importable so ``utils`` can be imported.
REPO_ROOT = "/Users/melih.gorgulu/Desktop/Projects/aftercourt_automation"
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from utils.prod_utils import get_data_by_attachment_id  # noqa: E402
from python_utilities.db_connection import DbConnection  # noqa: E402

AWS_PROFILE = "739275445236_DataScienceUser"


def main() -> None:
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)

    csv_files = sorted(glob.glob(os.path.join(CSV_DIR, "*.csv")))
    if not csv_files:
        print(f"No CSV files found in: {CSV_DIR}")
        return

    # Collect the union of attachment_ids across all dataset CSVs.
    attachment_ids = set()
    for csv_path in csv_files:
        df = pd.read_csv(csv_path)
        ids = df["attachment_id"].dropna().astype(int).unique().tolist()
        attachment_ids.update(ids)
        print(f"{os.path.basename(csv_path)}: {len(ids)} attachment(s)")

    attachment_ids = sorted(attachment_ids)
    print(f"\nTotal unique attachment(s) to download: {len(attachment_ids)}")

    analytics_db = DbConnection("ANALYTICS", "PROD_RDS")
    session = boto3.Session(profile_name=AWS_PROFILE)
    s3 = session.client("s3")

    for idx, attachment_id in enumerate(attachment_ids, 1):
        target = os.path.join(DOWNLOAD_DIR, f"{attachment_id}.pdf")
        if os.path.exists(target):
            print(f"[{idx}/{len(attachment_ids)}] {attachment_id} already downloaded, skipping.")
            continue
        try:
            get_data_by_attachment_id(
                attachment_id,
                analytics_db,
                s3,
                pdf_download=True,
                pdf_download_dir=DOWNLOAD_DIR,
                verbose=False,
            )
            print(f"[{idx}/{len(attachment_ids)}] downloaded {attachment_id}.pdf")
        except Exception as exc:  # noqa: BLE001
            print(f"[{idx}/{len(attachment_ids)}] FAILED {attachment_id}: {exc}")

    print(f"\nDone. PDFs saved to: {DOWNLOAD_DIR}")


if __name__ == "__main__":
    main()
