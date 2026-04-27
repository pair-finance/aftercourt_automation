"""
Shared MLflow helpers for the aftercourt-automation DVC pipeline.

Usage
-----
    from src.mlflow_utils import init_mlflow, get_or_create_run

Each DVC target (ladung, pfub, …) gets its own MLflow run.
Stages within the same target (train → evaluate) share one run via a
per-target run-id file: ``mlruns/.active_run_id.<target>``.
"""

import os
import logging
import yaml
import mlflow

logger = logging.getLogger(__name__)

_PARAMS_PATH = "params.yaml"
_ACTIVE_RUN_ID_PREFIX = "mlruns/.active_run_id"


def _load_mlflow_params() -> dict:
    """Read the ``mlflow`` section from params.yaml."""
    with open(_PARAMS_PATH) as f:
        return yaml.safe_load(f).get("mlflow", {})


def init_mlflow() -> str:
    """
    Configure the MLflow tracking URI and experiment from ``params.yaml``.

    Returns the experiment name.
    """
    cfg = _load_mlflow_params()
    tracking_uri = cfg.get("tracking_uri", "mlruns")
    experiment_name = cfg.get("experiment_name", "aftercourt_automation")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    logger.info("MLflow tracking URI : %s", tracking_uri)
    logger.info("MLflow experiment   : %s", experiment_name)
    return experiment_name


def get_or_create_run(stage_name: str, target: str, run_name: str | None = None):
    """
    Return an ``mlflow.start_run`` context manager.

    * If an active run ID is stored on disk for this *target* (from a previous
      stage in the same ``dvc repro``), that run is resumed so all stages
      within the same target share a single run.
    * Otherwise a new run is created and its ID is persisted for downstream
      stages of the same target.

    Parameters
    ----------
    stage_name : str
        DVC stage name – logged as a tag.
    target : str
        Pipeline target (e.g. "ladung", "pfub"). Each target gets its own
        run-id file so targets never interfere with each other.
    run_name : str, optional
        Human-readable run name shown in the UI.
    """
    run_id = _read_active_run_id(target)
    if run_id:
        logger.info("Resuming MLflow run %s for stage '%s' target '%s'", run_id, stage_name, target)
        run = mlflow.start_run(run_id=run_id)
    else:
        logger.info("Starting new MLflow run for stage '%s' target '%s'", stage_name, target)
        run = mlflow.start_run(run_name=run_name)
        _write_active_run_id(run.info.run_id, target)
    # Always (re-)set the run name so the latest stage's name wins even when
    # resuming a run that was created by an earlier stage.
    if run_name:
        mlflow.set_tag("mlflow.runName", run_name)
    mlflow.set_tag("dvc_stage", stage_name)
    mlflow.set_tag("target", target)
    return run


def finish_pipeline_run(target: str):
    """
    Remove the persisted active-run-id file for the given *target* so the
    *next* ``dvc repro`` starts a fresh MLflow run for that target.
    """
    run_id_file = f"{_ACTIVE_RUN_ID_PREFIX}.{target}"
    if os.path.exists(run_id_file):
        os.remove(run_id_file)
        logger.info("Cleared active MLflow run id file for target '%s'", target)


def log_params_flat(params: dict, prefix: str = ""):
    """
    Recursively flatten a nested dict and log every leaf value as an MLflow
    parameter.  Keys are dot-separated (e.g. ``prepare.ladung.target_col``).
    """
    for key, value in params.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            log_params_flat(value, prefix=full_key)
        else:
            mlflow.log_param(full_key, value)


# ── internal helpers ─────────────────────────────────────────────────────

def _read_active_run_id(target: str) -> str | None:
    run_id_file = f"{_ACTIVE_RUN_ID_PREFIX}.{target}"
    if os.path.exists(run_id_file):
        with open(run_id_file) as f:
            run_id = f.read().strip()
        return run_id if run_id else None
    return None


def _write_active_run_id(run_id: str, target: str):
    run_id_file = f"{_ACTIVE_RUN_ID_PREFIX}.{target}"
    os.makedirs(os.path.dirname(run_id_file), exist_ok=True)
    with open(run_id_file, "w") as f:
        f.write(run_id)
    logger.info("Persisted active MLflow run id for target '%s': %s", target, run_id)
