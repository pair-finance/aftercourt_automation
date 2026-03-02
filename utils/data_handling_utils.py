import uuid
import hashlib
import pandas as pd


def generate_uuid(text: str, purpose: str) -> str:
    """
    Generate a UUID based on the input text.

    Args:
        text (str): The input text to generate the UUID from.
        purpose (str): The purpose for which the UUID is being generated.


    Returns:
        str: A UUID string generated from the input text and purpose.
    """
    text = text[:100]  # Limit the text to the first 100 characters
    signature = f"{purpose}:{text}"
    # Use nil UUID as namespace (compatible with all Python versions)
    nil_namespace = uuid.UUID('00000000-0000-0000-0000-000000000000')
    return str(uuid.uuid5(nil_namespace, signature))


def generate_hash_from_text(text: str) -> str:
    text = text[:100]  # Limit the text to the first 100 characters
    """Generate SHA-256 hash from text string"""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def extract_filename_from_object_key(object_key: str) -> str:
    if pd.isna(object_key):
        return None
    return object_key.split('/')[-1]

def extract_s3_key(row):
    created_at = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    base = "ocr_source_files"
    date_folder = created_at.split(' ')[0]  # Extract date part: YYYY-MM-DD
    egvp_id = row["egvp_id"]
    egvp_folder = f"egvp_id_{egvp_id}"
    full_path = f"{base}/{date_folder}/{egvp_folder}/{row['file_name']}"
    return full_path