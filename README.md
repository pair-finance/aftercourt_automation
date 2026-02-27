# aftercourt_automation
A repository to track works for Aftercourt Automation

## Experiment Tracking with MLflow

The DVC pipeline stages (`train` and `evaluate`) automatically log parameters,
metrics, artifacts, and plots to **MLflow**.

### Quick start

```bash
# 1. Install MLflow (already in environment.yaml)
pip install "mlflow>=2.12.0"

# 2. Run the full pipeline – MLflow logging happens automatically
dvc repro

# 3. Launch the MLflow UI to browse experiments
mlflow ui --backend-store-uri mlruns
# Then open http://127.0.0.1:5000 in your browser
```

### Configuration

MLflow settings live in `params.yaml` under the `mlflow` key:

```yaml
mlflow:
  tracking_uri: mlruns     # local directory (default)
  experiment_name: aftercourt_automation
```

To switch to a **remote tracking server**, change `tracking_uri`:

```yaml
mlflow:
  tracking_uri: http://mlflow-server:5000
  experiment_name: aftercourt_automation
```

### What gets logged

| Stage      | Logged to MLflow                                                           |
|------------|---------------------------------------------------------------------------|
| **train**  | All pipeline params, training set stats, sklearn model (model registry)   |
| **evaluate** | Accuracy, precision, recall, F1, ROC-AUC, confusion matrix, threshold, ROC/PR/CM plots |

Both stages share a **single MLflow run** per `dvc repro` invocation, so all
information is visible in one place.

### Directory layout

- `mlruns/` – local MLflow tracking store (git-ignored)
- `src/mlflow_utils.py` – shared helpers (`init_mlflow`, `get_or_create_run`, …)
