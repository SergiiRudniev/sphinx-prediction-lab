# Repository Instructions

## Research Discipline

- Register a stable `SPH-T-H###` ID before running an experiment.
- Never use test or forward labels for architecture, checkpoint, threshold or policy selection.
- Preserve point-in-time availability for every market, wallet and graph feature.
- Compute wallet reputation only from outcomes resolved before the decision timestamp.
- Record rejected and invalidated results; do not rewrite the journal from memory.
- Never describe a wallet as an insider. Use evidence-bounded informed-flow language.

## Engineering

- Keep model recommendations separate from deterministic execution and risk controls.
- Do not enable live trading in repository defaults.
- Do not commit raw datasets, checkpoints, credentials, signed orders or private keys.
- Update machine-readable contracts together with architecture changes.
- Add tests for causal boundaries, schemas and policy behavior.

## Quality

Run before committing:

```bash
python scripts/check_contracts.py
ruff check .
mypy src tests scripts
pytest
```
