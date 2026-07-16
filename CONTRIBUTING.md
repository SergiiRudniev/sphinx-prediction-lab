# Contributing

Sphinx Prediction Lab accepts changes that preserve causal evidence and keep model
analysis separate from capital controls.

## Workflow

1. Open a research hypothesis or issue before material model/data changes.
2. Create a focused branch from `main`.
3. Update contracts, documentation and tests together.
4. Open a pull request using the repository template.
5. Do not open untouched labels until the registered protocol permits it.

## Local Checks

```bash
python -m pip install -e ".[dev]"
python scripts/check_contracts.py
ruff check .
mypy src tests scripts
pytest
```

## Research Results

Every executed hypothesis must append its result to `docs/RESEARCH.md`, including
failures. Include source and config hashes, split boundaries, seeds, label-open
status and the final decision.

## Data

Do not submit raw third-party datasets. Submit schemas, manifests, deterministic
builders and small synthetic fixtures with no real credentials or private material.
