"""
textract_utils.py

Utility module for interacting with AWS Textract, an OCR service that extracts
text from documents stored in S3. This module provides:

- Configuration dataclass (ModelConfig) for Textract job settings.
- A boto3-backed TextractClient wrapper for submitting and polling Textract jobs.
- Helper functions for batch-submitting jobs, waiting for completion, and
  converting raw Textract block outputs into plain text strings.
- A high-level pipeline function (aws_textract_pipeline) that processes lists
  of S3 object keys in batches.
- parse_pdfs_with_textract as the primary entry point for callers that need
  extracted text keyed by S3 object path.
"""

import awswrangler.secretsmanager as sm
import boto3
import json
import logging
import os
import requests
from botocore.exceptions import ClientError
from typing import Dict, List, Iterable, Tuple

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_secret(secret_id, value):
    """Retrieve a single value from an AWS Secrets Manager JSON secret.

    Args:
        secret_id (str): The ARN or name of the secret in Secrets Manager.
        value (str): The key to look up inside the secret's JSON payload.

    Returns:
        Any: The value associated with ``value`` in the secret, or ``None``
        if the key does not exist.
    """
    return sm.get_secret_json(
        secret_id,
        boto3_session=boto3.Session(region_name="eu-central-1"),
    ).get(value)
        

class ModelConfig:
    """Static configuration constants for Textract OCR jobs.

    Attributes:
        lost_condition_interval_days (int): Number of days after which an OCR
            job is considered lost and eligible for a retry.
        scheduler_interval_lost_llm_hours (int): Polling interval (hours) for
            the scheduler that checks for lost LLM jobs.
        scheduler_interval_textract_send_minutes (int): Interval (minutes)
            between batches of Textract job submissions.
        scheduler_interval_textract_get_minutes (int): Interval (minutes)
            between Textract result retrieval runs.
        ocr_s3_bucket (str): S3 bucket used for OCR source files and results.
        ocr_source_path (str): S3 prefix for uploaded source documents.
        ocr_results_path (str): S3 prefix for prepared OCR output files.
        irrelevant_file_extentions (List[str]): File extensions that should be
            silently ignored (not submitted to Textract).
        unsupported_file_extentions (List[str]): File extensions that are
            explicitly unsupported by the pipeline.
    """

    lost_condition_interval_days = 7
    scheduler_interval_lost_llm_hours = 2
    scheduler_interval_textract_send_minutes = 10
    scheduler_interval_textract_get_minutes = 10
    ocr_s3_bucket = "pair-email-classification"
    ocr_source_path = "ocr_source_files"
    ocr_results_path = "ocr_prepared_output"
    irrelevant_file_extentions = ["vcf", "gif", "p7s", "asc", "ics", "bin"]
    unsupported_file_extentions = ["html", "mp4"]
    


class TextractClient:
    """Thin wrapper around the boto3 Textract client.

    Handles job submission, status polling, and paginated result retrieval
    for asynchronous document text detection jobs.
    """

    def __init__(self, model_config):
        """Initialise the client with the given configuration.

        Args:
            model_config (ModelConfig): Configuration object that supplies
                bucket names, path prefixes, and other job parameters.
        """
        self.model_config = model_config
        self.boto3_session = boto3.Session(
            region_name="eu-central-1",
            profile_name="739275445236_DataScienceUser"
        )
        self.client = self.boto3_session.client("textract")

    def submit_textract_job(self, bucket_name, document_key):
        """Start an asynchronous Textract text-detection job.

        Args:
            bucket_name (str): Name of the S3 bucket containing the document.
            document_key (str): S3 object key of the document to process.

        Returns:
            str | None: The Textract ``JobId`` on success, or ``None`` if the
            submission failed.
        """
        try:
            response = self.client.start_document_text_detection(
                DocumentLocation={
                    "S3Object": {"Bucket": bucket_name, "Name": document_key}
                }
            )
            return response["JobId"]
        except Exception as e:
            print(
                f"Error submitting Textract job for document {document_key}: {str(e)}"
            )
            return None

    def check_job_status(self, job_id) -> Dict:
        """Poll Textract for the current status of a job.

        If the job has succeeded, the full block-level results are fetched
        and included in the returned dictionary.

        Args:
            job_id (str): The Textract job ID to query.

        Returns:
            Dict: A dict with at least a ``'status'`` key (``'IN_PROGRESS'``,
            ``'SUCCEEDED'``, ``'FAILED'``, or ``'ERROR'``). When status is
            ``'SUCCEEDED'``, an additional ``'result'`` key contains the list
            of Textract blocks.
        """
        try:
            response = self.client.get_document_text_detection(JobId=job_id)
            status = response["JobStatus"]

            if status == "SUCCEEDED":
                result = self.get_job_results(job_id)
                return {"status": status, "result": result}
            else:
                return {"status": status}

        except Exception as e:
            print(f"Error checking status for job {job_id}: {str(e)}")
            return {"status": "ERROR"}

    def get_job_results(self, job_id) -> List:
        """Retrieve all result blocks for a completed Textract job.

        Handles pagination transparently so the caller receives the full set
        of blocks regardless of how many pages the response spans.

        Args:
            job_id (str): The Textract job ID of a completed job.

        Returns:
            List[Dict]: A flat list of all Textract block objects returned by
            the service.
        """
        results = []
        pagination_token = None
        while True:
            if pagination_token:
                response = self.client.get_document_text_detection(
                    JobId=job_id, NextToken=pagination_token
                )
            else:
                response = self.client.get_document_text_detection(JobId=job_id)
            # Note: we could also get document metedata to get info like # of pages etc.
            results.extend(response["Blocks"])

            if "NextToken" in response:
                pagination_token = response["NextToken"]
            else:
                break

        return results
    

def _get_textract_client() -> TextractClient:
    """Create and return a :class:`TextractClient` with default configuration.

    Returns:
        TextractClient: A ready-to-use client configured with
        :class:`ModelConfig` defaults.
    """
    model_config = ModelConfig()
    return TextractClient(model_config)

from typing import List, Dict, Any
import time


def get_texts_with_page_markers_from_textract_outputs(
    textract_outputs: List[Dict[str, Any]],
) -> List[str]:
    """Convert raw Textract block lists into plain text with page markers.

    Like :func:`get_texts_from_textract_outputs`, but preserves page
    boundaries by prefixing each page's text with a ``<page_N>`` marker.

    Args:
        textract_outputs (List[Dict[str, Any] | None]): A list where each
            element is either a list of Textract block dicts (as returned by
            :meth:`TextractClient.get_job_results`) or ``None``.

    Returns:
        List[str]: A list of extracted text strings, one per input document.
        Each page within a document is preceded by a ``<page_N>`` marker on
        its own line. Failed or empty documents produce an empty string.
    """
    from collections import defaultdict

    raw_text_outputs: List[str] = []
    for output in textract_outputs:
        if output is None:
            raw_text_outputs.append("")
            continue

        pages_to_lines: Dict[int, List[str]] = defaultdict(list)
        for block in output:
            if block["BlockType"] == "LINE":
                page_number = int(block.get("Page", 1))
                pages_to_lines[page_number].append(block["Text"])

        parts = [
            f"<page_{page}>\n" + "\n".join(pages_to_lines[page])
            for page in sorted(pages_to_lines)
        ]
        raw_text_outputs.append("\n".join(parts))

    return raw_text_outputs


def get_texts_from_textract_outputs(textract_outputs: List[Dict[str, Any]]) -> List[str]:
    """Convert raw Textract block lists into plain text strings.

    Each element in ``textract_outputs`` corresponds to one document. ``None``
    entries (e.g. for failed or skipped documents) are converted to empty
    strings. For valid block lists, only ``LINE``-type blocks are concatenated
    to form the document text.

    Args:
        textract_outputs (List[Dict[str, Any] | None]): A list where each
            element is either a list of Textract block dicts (as returned by
            :meth:`TextractClient.get_job_results`) or ``None``.

    Returns:
        List[str]: A list of extracted text strings, one per input document.
        Failed or empty documents produce an empty string ``""``.
    """
    raw_text_outputs = []
    for output in textract_outputs:
        if output is None:
            raw_text_outputs.append("")
        else:
            raw_text = []
            for cur_doc in output:
                if cur_doc["BlockType"] == "LINE":
                    raw_text.append(cur_doc["Text"])    
               
            raw_text_outputs.append("\n".join(raw_text))

    return raw_text_outputs

def submit_textract_job(textract_client: TextractClient, doc_key: str) -> Dict[str, Any]:
    """Submit a single Textract job and return job info."""
    try:
        job_id = textract_client.submit_textract_job(
            bucket_name=textract_client.model_config.ocr_s3_bucket, 
            document_key=doc_key
        )
        logger.info("Textract job submitted for document '%s' with job_id '%s'", doc_key, job_id)
        return {
            'job_id': job_id,
            'doc_key': doc_key,
            'status': 'SUBMITTED'
        }
    except Exception as e:
        logger.error("Failed to submit job for document '%s': %s", doc_key, str(e))
        return {
            'job_id': None,
            'doc_key': doc_key,
            'status': 'FAILED',
            'error': str(e)
        }


def wait_for_job_completion(textract_client: TextractClient, job_info: Dict[str, Any], max_wait_time: int = 100) -> Dict[str, Any]:
    """Wait for a Textract job to complete and return the result."""
    job_id = job_info['job_id']
    doc_key = job_info['doc_key']
    
    if job_info['status'] == 'FAILED':
        return {
            'doc_key': doc_key,
            'result': None,
            'status': 'FAILED',
            'error': job_info.get('error', 'Job submission failed')
        }
    
    max_count = 0
    while max_count < max_wait_time:
        try:
            time.sleep(1)
            res = textract_client.check_job_status(job_id=job_id)
            
            if res["status"] == "SUCCEEDED":
                logger.info("Job Completed for job_id '%s' (document: %s)", job_id, doc_key)
                return {
                    'doc_key': doc_key,
                    'result': res['result'],
                    'status': 'SUCCEEDED'
                }
            elif res["status"] == "FAILED":
                logger.error("Job Failed for job_id '%s' (document: %s)", job_id, doc_key)
                return {
                    'doc_key': doc_key,
                    'result': None,
                    'status': 'FAILED',
                    'error': 'Textract job failed'
                }
                
        except Exception as e:
            logger.error("Error checking job status for job_id '%s': %s", job_id, str(e))
            
        max_count += 1
    
    # Timeout reached
    logger.warning("Max retries exceeded for job_id '%s' (document: %s)", job_id, doc_key)
    
    return {
        'doc_key': doc_key,
        'result': None,
        'status': 'TIMEOUT',
        'error': 'Job timeout'
    }



def aws_textract_pipeline(object_keys: List[str], max_workers: int = 10) -> List:
    """Run a batch Textract OCR pipeline over a list of S3 object keys.

    Documents are processed in fixed-size batches. Each batch is fully
    submitted before the pipeline waits for individual job completions,
    improving throughput compared to a purely sequential approach.

    Args:
        object_keys (List[str]): S3 object keys of the documents to process.
            Keys must reside in the bucket defined by
            :attr:`ModelConfig.ocr_s3_bucket`.
        max_workers (int): Currently unused; reserved for future parallel
            execution support. Defaults to ``10``.

    Returns:
        List[List[Dict] | None]: A list of raw Textract block lists in the
        same order as ``object_keys``. Entries for documents that failed or
        timed out are ``None``.
    """

    textract_outputs = [None] * len(object_keys)  # Placeholder for results
    
    logger.info("Starting AWS Textract Jobs with batch processing (max {} jobs at a time)", max_workers)
    batch_size = 25
    job_results = []
    total_jobs = len(object_keys)
    submitted = 0
    
    textract_client: TextractClient = _get_textract_client()

    while submitted < total_jobs:
        current_batch_keys = object_keys[submitted:submitted+batch_size]
        job_infos = []
        # Submit jobs for current batch
        for doc_key in current_batch_keys:
            job_info = submit_textract_job(textract_client, doc_key)
            job_infos.append(job_info)
            
        logger.info(f"Submitted batch {submitted//batch_size+1}: {len(current_batch_keys)} jobs")
        
        # Wait for jobs in current batch
        batch_results = []
        for job_info in job_infos:
            result = wait_for_job_completion(textract_client, job_info)
            batch_results.append(result)
    
        job_results.extend(batch_results)
        submitted += batch_size
        
        logger.info(f"Completed batch {submitted//batch_size}: {min(submitted, total_jobs)}/{total_jobs} jobs")
        
    # Sort results to maintain original order
    doc_key_to_result = {result['doc_key']: result for result in job_results}
    ordered_results = [doc_key_to_result[doc_key] for doc_key in object_keys]
    # Process results and update cache and output array
    successful_count = 0
    failed_count = 0
    
    for result in ordered_results:
        original_index = object_keys.index(result['doc_key'])
        if result['status'] == 'SUCCEEDED':
            textract_outputs[original_index] = result['result']
            successful_count += 1
        else:
            textract_outputs[original_index] = None
            failed_count += 1
            logger.warning("Failed to process document '%s': %s", result['doc_key'], result.get('error', 'Unknown error'))
            
    logger.info("AWS Textract pipeline completed: %d successful, %d failed", successful_count, failed_count)


    # Final validation - ensure no None values where we expect results
    total_successful = sum(1 for result in textract_outputs if result is not None)
    total_failed = len(textract_outputs) - total_successful

    logger.info("Final results: %d successful, %d failed (including cached)", total_successful, total_failed)

    return textract_outputs


def parse_pdfs_with_textract(object_keys: List[str]) -> List[str]:
    """High-level entry point: extract text from S3 documents using Textract.

    Runs the full Textract pipeline and returns a mapping from each S3 object
    key to its extracted plain-text content.

    Args:
        object_keys (List[str]): S3 object keys of the PDF (or image) files to
            process.

    Returns:
        Dict[str, str]: A dictionary mapping each object key to its extracted
        text string. Documents that could not be processed map to an empty
        string ``""``.
    """
    textract_outputs = aws_textract_pipeline(object_keys, max_workers=2)
    raw_text_outputs = get_texts_from_textract_outputs(textract_outputs)
    new_texts_dict = {obj_key: text for obj_key, text in zip(object_keys, raw_text_outputs)}
    return new_texts_dict

def parse_pdfs_with_pagemarkers_with_textract(object_keys: List[str]) -> List[str]:
    """High-level entry point: extract text from S3 documents using Textract.

    Runs the full Textract pipeline and returns a mapping from each S3 object
    key to its extracted plain-text content.

    Args:
        object_keys (List[str]): S3 object keys of the PDF (or image) files to
            process.

    Returns:
        Dict[str, str]: A dictionary mapping each object key to its extracted
        text string. Documents that could not be processed map to an empty
        string ``""``.
    """
    textract_outputs = aws_textract_pipeline(object_keys, max_workers=2)
    raw_text_outputs_w_pagemarkers = get_texts_with_page_markers_from_textract_outputs(textract_outputs)
    new_texts_dict = {obj_key: text for obj_key, text in zip(object_keys, raw_text_outputs_w_pagemarkers)}
    return new_texts_dict


def parse_local_pdfs_with_textract(
    local_pdf_paths: List[str],
    s3_object_key_base: str,
    s3_bucket_name: str = ModelConfig.ocr_s3_bucket,
    use_page_markers: bool = False,
) -> Dict[str, str]:
    """Parse local PDF files using AWS Textract and return object key to text mapping.

    Uploads each local PDF to S3 under the given prefix, then runs the Textract
    pipeline to extract text from each document.

    Args:
        local_pdf_paths: List of absolute paths to local PDF files.
        s3_object_key_base: S3 key prefix under which files will be uploaded
            (e.g. "data/pfub_invoices/").
        s3_bucket_name: Target S3 bucket. Defaults to ModelConfig.ocr_s3_bucket.

    Returns:
        Dict mapping each S3 object key to the extracted text string.
    """
    session = boto3.Session(
        region_name="eu-central-1",
        profile_name="739275445236_DataScienceUser",
    )
    s3_client = session.client("s3")

    object_keys = []
    for pdf_path in local_pdf_paths:
        file_name = os.path.basename(pdf_path)
        object_key = os.path.join(s3_object_key_base, file_name)
        logger.info("Uploading '%s' to s3://%s/%s", pdf_path, s3_bucket_name, object_key)
        s3_client.upload_file(pdf_path, s3_bucket_name, object_key)
        object_keys.append(object_key)

    return parse_pdfs_with_textract(object_keys) if not use_page_markers else parse_pdfs_with_pagemarkers_with_textract(object_keys)


def _s3_find_existing_key(s3_client, bucket: str, key_base: str, attachment_id: str):
    """Return the existing S3 key for ``attachment_id`` (any extension), or None.

    Matches by the stable ``attachment_id`` rather than a full key with a
    URL-derived extension, so the upload-skip works even when the source
    (token) URL or its ``name=`` parameter changes between runs. An object is
    considered a match only when its filename (without extension) is exactly
    ``attachment_id`` - this avoids ``16801640-1`` matching ``16801640-10``.
    """
    prefix = os.path.join(key_base, attachment_id)
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if os.path.splitext(os.path.basename(key))[0] == attachment_id:
                return key
    return None


def _s3_object_exists(s3_client, bucket: str, key: str) -> bool:
    """Return True if an object already exists at ``s3://bucket/key``."""
    try:
        s3_client.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] in ("404", "NoSuchKey", "NotFound"):
            return False
        raise


def parse_url_attachments_with_textract(
    attachments: Iterable[Tuple[str, str]],
    s3_object_key_base: str,
    s3_bucket_name: str = ModelConfig.ocr_s3_bucket,
    use_page_markers: bool = False,
    skip_if_exists: bool = True,
    skip_upload: bool = False,
    request_timeout: int = 60,
    cache_path: str = None,
    batch_size: int = 100,
) -> Dict[str, str]:
    """Parse attachments from token URLs using Textract, streaming bytes straight to S3.

    Unlike :func:`parse_local_pdfs_with_textract`, this avoids the local-disk
    round-trip: each attachment is downloaded from its (pre-signed/token) URL
    and uploaded to S3 in memory. S3 itself acts as the upload cache - when
    ``skip_if_exists`` is True, an attachment whose object key already exists
    in the bucket is not re-uploaded.

    Textract *results* are additionally cached to a local JSON file
    (``cache_path``) keyed by S3 object key, so re-runs skip already-OCR'd
    documents and avoid re-incurring Textract cost/time. Results are persisted
    after each batch.

    Args:
        attachments: Iterable of ``(attachment_id, url)`` pairs. The
            ``attachment_id`` is used to build a stable S3 object key.
        s3_object_key_base: S3 key prefix under which files are uploaded
            (e.g. "ocr_source_files/letters").
        s3_bucket_name: Target S3 bucket. Defaults to ModelConfig.ocr_s3_bucket.
        use_page_markers: If True, returned text includes ``<page_N>`` markers.
        skip_if_exists: If True, skip the upload when the object already exists.
        skip_upload: If True, never download from URLs or upload to S3. Assumes
            every attachment is already present in S3; object keys are resolved
            by looking up the existing S3 object for each ``attachment_id``.
        request_timeout: Per-request download timeout in seconds.
        cache_path: Optional path to a JSON file used to cache Textract results
            keyed by S3 object key. If None, no result caching is performed.
        batch_size: Number of documents to OCR per batch before persisting the
            cache.

    Returns:
        Dict mapping each S3 object key to the extracted text string, covering
        all input attachments (from cache and freshly parsed).
    """
    session = boto3.Session(
        region_name="eu-central-1",
        profile_name="739275445236_DataScienceUser",
    )
    s3_client = session.client("s3")

    # Load Textract-result cache.
    textract_cache: Dict[str, str] = {}
    if cache_path and os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            textract_cache = json.load(f)
    logger.info("[Textract Cache] Loaded %d cached entries", len(textract_cache))

    # Upload to S3 (skipping cached/existing) and collect object keys to parse.
    attachments = list(attachments)
    total_attachments = len(attachments)
    logger.info("[Upload] Starting upload phase for %d attachments", total_attachments)

    object_keys: List[str] = []
    keys_to_parse: List[str] = []
    uploaded = 0
    upload_skipped = 0
    cache_hits = 0
    download_failed = 0
    progress_every = 25  # emit a progress line at least this often

    # When skipping the upload entirely, resolve object keys from the existing
    # Textract cache (mapping attachment_id -> cached key) so we never hit S3.
    cache_key_by_attachment_id: Dict[str, str] = {}
    if skip_upload:
        for cached_key in textract_cache:
            cache_key_by_attachment_id[os.path.splitext(os.path.basename(cached_key))[0]] = cached_key

    for idx, (attachment_id, url) in enumerate(attachments, start=1):
        # Resolve the S3 key. If a file for this attachment_id already exists in
        # S3 (regardless of extension), reuse that exact key so we can skip the
        # upload. This makes the skip robust to URL/extension changes between runs.
        if skip_upload:
            # No S3 calls: prefer the cached key, otherwise build it from the URL.
            existing_key = cache_key_by_attachment_id.get(attachment_id)
        else:
            existing_key = (
                _s3_find_existing_key(s3_client, s3_bucket_name, s3_object_key_base, attachment_id)
                if skip_if_exists else None
            )
        if existing_key is not None:
            object_key = existing_key
        else:
            ext = os.path.splitext(url.split("name=")[-1])[-1] if "name=" in url else ".pdf"
            object_key = os.path.join(s3_object_key_base, f"{attachment_id}{ext}")
        object_keys.append(object_key)

        # Already OCR'd -> no need to upload or re-parse.
        if object_key in textract_cache:
            cache_hits += 1
        else:
            keys_to_parse.append(object_key)

            if skip_upload or existing_key is not None:
                upload_skipped += 1
                logger.info(
                    "[Upload] Skipped '%s' (already in S3; %d skipped so far)",
                    object_key, upload_skipped,
                )
            else:
                resp = requests.get(url, timeout=request_timeout)
                if resp.status_code != 200:
                    download_failed += 1
                    logger.warning(
                        "[Upload] Download failed for attachment '%s' (HTTP %s)",
                        attachment_id, resp.status_code,
                    )
                else:
                    s3_client.put_object(Bucket=s3_bucket_name, Key=object_key, Body=resp.content)
                    uploaded += 1
                    logger.info(
                        "[Upload] Uploaded '%s' (%d uploaded so far)", object_key, uploaded
                    )

        # Periodic running summary so progress/remaining is always visible.
        if idx % progress_every == 0 or idx == total_attachments:
            processed = idx
            remaining = total_attachments - processed
            logger.info(
                "[Upload] Progress %d/%d (%.1f%%) | uploaded=%d, already_in_s3=%d, "
                "cache_hits=%d, download_failed=%d, remaining=%d",
                processed, total_attachments, 100.0 * processed / max(total_attachments, 1),
                uploaded, upload_skipped, cache_hits, download_failed, remaining,
            )

    logger.info(
        "[Upload] Complete: %d uploaded, %d already in S3, %d cache hits, %d download failed; "
        "%d to OCR out of %d total",
        uploaded, upload_skipped, cache_hits, download_failed, len(keys_to_parse), total_attachments,
    )

    # OCR remaining documents in batches, persisting the cache after each batch.
    parse_fn = parse_pdfs_with_pagemarkers_with_textract if use_page_markers else parse_pdfs_with_textract
    for batch_start in range(0, len(keys_to_parse), batch_size):
        batch = keys_to_parse[batch_start:batch_start + batch_size]
        batch_num = batch_start // batch_size + 1
        total_batches = (len(keys_to_parse) + batch_size - 1) // batch_size
        logger.info("[Textract] Processing batch %d/%d (%d files)", batch_num, total_batches, len(batch))

        new_results = parse_fn(batch)
        textract_cache.update(new_results)

        if cache_path:
            with open(cache_path, "w") as f:
                json.dump(textract_cache, f)
            logger.info("[Textract] Cached batch %d/%d, total cached: %d", batch_num, total_batches, len(textract_cache))

    # Return results for all requested attachments (cache + freshly parsed).
    return {key: textract_cache.get(key, "") for key in object_keys}


def text_from_s3_link(s3_link: str) -> str:
    import json
    import boto3

    session = boto3.Session(
        region_name="eu-central-1",
        profile_name="739275445236_DataScienceUser",
    )
    s3_client = session.client("s3")

    """Download a Textract-blocks JSON from S3 and merge LINE blocks into text."""
    bucket, key = s3_link.replace("s3://", "").split("/", 1)
    obj = s3_client.get_object(Bucket=bucket, Key=key)
    blocks = json.loads(obj["Body"].read())
    if isinstance(blocks, dict):
        blocks = blocks.get("Blocks", [])
    return "\n".join(b["Text"] for b in blocks if b.get("BlockType") == "LINE")
