
import json
from datetime import date, timedelta
from typing import Optional

import boto3

session = boto3.Session(profile_name="739275445236_DataScienceUser")
s3 = session.client("s3")

bucket = "pair-scanner"
prefix_base = "ocr-v3/egvp/"


def search_egvp_id_in_exported_and_archived(
    egvp_id: str,
    start_date: date,
    end_date: date,
) -> dict:
    """Search for an EGVP ID in both exported and archived S3 data for the given date range.

    Args:
        egvp_id: The EGVP message ID to search for.
        start_date: Start date of the search range (inclusive).
        end_date: End date of the search range (inclusive).

    Returns:
        A dict with keys:
            - found_in_exported (bool)
            - found_in_archived (bool)
            - exported_content (Optional[dict]): The message dict from exported data, if found.
            - archived_content (Optional[dict]): The message dict from archived data, if found.
    """
    result = {
        "found_in_exported": False,
        "found_in_archived": False,
        "exported_content": None,
        "archived_content": None,
    }

    # --- Search exported data ---
    current = start_date
    while current <= end_date:
        date_folder = current.strftime("%Y_%m_%d")
        prefix = f"{prefix_base}{date_folder}/"
        response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
        for obj in response.get("Contents", []):
            if "export" not in obj["Key"]:
                continue
            body = s3.get_object(Bucket=bucket, Key=obj["Key"])["Body"].read()
            exported_data = json.loads(body)
            for message_dict in exported_data:
                if message_dict.get("messageId") == egvp_id:
                    result["found_in_exported"] = True
                    result["exported_content"] = message_dict
                    break
            if result["found_in_exported"]:
                break
        if result["found_in_exported"]:
            break
        current += timedelta(days=1)

    # --- Search archived data ---
    archive_prefix = "ocr-v3/data/archive/"
    date_prefixes = set()
    current = start_date
    while current <= end_date:
        date_prefixes.add(current.strftime("%Y%m%d"))
        current += timedelta(days=1)

    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=archive_prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            filename = key.rsplit("/", 1)[-1]
            if not filename.endswith(".json"):
                continue
            if not any(filename.startswith(dp) for dp in date_prefixes):
                continue
            body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
            archive_data = json.loads(body)
            items = archive_data if isinstance(archive_data, list) else [archive_data]
            for msg in items:
                if not isinstance(msg, dict):
                    continue
                msg_id = msg.get("messageId") or msg.get("id") or msg.get("egvpId")
                if msg_id == egvp_id:
                    result["found_in_archived"] = True
                    result["archived_content"] = msg
                    break
            if result["found_in_archived"]:
                break
        if result["found_in_archived"]:
            break

    return result

if __name__ == "__main__":
    no_data_in_db_ids = [
        "NRW_B21779880137380aa247fae-d9da-414e-930a-29e8a89fbbe8",
        "NRW_B21779878609522212a62a3-48ac-4204-b804-1e9631d1ef9c",
        "NRW_B21779876441224117f26b7-4ce4-420a-96f8-ab88d6db00fa",
        "NRW_B217798765963076f897f9b-8c9c-4a8d-a03d-27c8385af8d9",
        "NRW_B217798769463350961754c-975b-41fe-bd0e-145e3e157917",
        "NRW_B21779871446660a621ebf1-854d-4f1d-997b-5393f7845be9",
        "NRW_B217798701706015a4f3e97-72ec-4ad9-a3bb-5026e714d942",
    ]
    start = date(2026, 5, 1)
    end = date(2026, 6, 1)
    result = {}
    for egvp_id in no_data_in_db_ids:
        print(f"Searching for EGVP ID: {egvp_id}")
        search_result = search_egvp_id_in_exported_and_archived(egvp_id, start, end)
        result[egvp_id] = search_result
    result_path = "egvp_id_search_results.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"Search completed. Results saved to {result_path}")
    