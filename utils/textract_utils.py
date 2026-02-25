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
import logging
from typing import Dict, List

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