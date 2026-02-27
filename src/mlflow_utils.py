"""
Shared MLflow helpers for the aftercourt-automation DVC pipeline.

Usage
-----
    from src.mlflow_utils import init_mlflow, get_or_create_run

Every DVC stage that wants to participate in the *same* MLflow run should:

1. Call ``init_mlflow()`` once at the beginning (sets tracking URI + experiment).
2. Call ``get_or_create_run(stage_name)`` to obtain an ``mlflow.ActiveRun`` context
   manager that either resumes the run written to ``mlruns/.active_run_id`` or
   starts a fresh one.  This way the train → evaluate stages share one run.
"""

import os
import logging
import yaml
import mlflow

logger = logging.getLogger(__name__)

_PARAMS_PATH = "params.yaml"
_ACTIVE_RUN_ID_FILE = "mlruns/.active_run_id"


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


def get_or_create_run(stage_name: str, run_name: str | None = None):
    """
    Return an ``mlflow.start_run`` context manager.

    * If an active run ID is stored on disk (from a previous stage in the same
      ``dvc repro``), that run is resumed so all stages share a single run.
    * Otherwise a new run is created and its ID is persisted for downstream
      stages.

    Parameters
    ----------
    stage_name : str
        DVC stage name – logged as a tag.
    run_name : str, optional
        Human-readable run name shown in the UI.
    """
    run_id = _read_active_run_id()
    if run_id:
        logger.info("Resuming MLflow run %s for stage '%s'", run_id, stage_name)
        run = mlflow.start_run(run_id=run_id)
    else:
        logger.info("Starting new MLflow run for stage '%s'", stage_name)
        run = mlflow.start_run(run_name=run_name)
        _write_active_run_id(run.info.run_id)
    # Always (re-)set the run name so the latest stage's name wins even when
    # resuming a run that was created by an earlier stage.
    if run_name:
        mlflow.set_tag("mlflow.runName", run_name)
    mlflow.set_tag("dvc_stage", stage_name)
    return run


def finish_pipeline_run():
    """
    Remove the persisted active-run-id file so the *next* ``dvc repro``
    starts a fresh MLflow run.
    """
    if os.path.exists(_ACTIVE_RUN_ID_FILE):
        os.remove(_ACTIVE_RUN_ID_FILE)
        logger.info("Cleared active MLflow run id file")


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

def _read_active_run_id() -> str | None:
    if os.path.exists(_ACTIVE_RUN_ID_FILE):
        with open(_ACTIVE_RUN_ID_FILE) as f:
            run_id = f.read().strip()
        return run_id if run_id else None
    return None


def _write_active_run_id(run_id: str):
    os.makedirs(os.path.dirname(_ACTIVE_RUN_ID_FILE), exist_ok=True)
    with open(_ACTIVE_RUN_ID_FILE, "w") as f:
        f.write(run_id)
    logger.info("Persisted active MLflow run id: %s", run_id)
