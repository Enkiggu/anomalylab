# AnomalyLab

AnomalyLab is a reproducible security telemetry analysis platform focused on anomaly detection, data quality, evaluation, and operational monitoring.

## Published foundation

The current foundation provides:

- deterministic synthetic security telemetry generation;
- typed dataset manifests and anomaly labels;
- data quality checks for schema, ranges, nulls, duplicates, and time ordering;
- stable dataset fingerprinting for reproducible experiments;
- versioned configurations for two telemetry scenarios.

## Development

Requirements:

- Python 3.12+

Install the project dependencies and run the tests:

```bash
python -m pip install -e .
python -m pytest -q
```

Generated datasets, experiment databases, model artifacts, logs, environment files, and dependency directories are intentionally excluded from version control.
