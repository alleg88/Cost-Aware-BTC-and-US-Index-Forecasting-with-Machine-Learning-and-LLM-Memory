# Experiments

Experiment modules fit models, build predictions and policies, verify frozen artifacts and
render thin reader notebooks. The main public entry points are:

| Command | Purpose |
|---|---|
| `python -m experiments.reproduce_tracked` | Verify the tracked checkout |
| `python -m experiments.reproduce_source --audit-only` | Audit the registered dependency graph |
| `python -m experiments.reproduce_source` | Rebuild registered numerical outputs |
| `python -m experiments.reproduce_notebooks` | Re-execute the 27 canonical readers |

Individual experiment modules remain available for bounded reruns and are registered by
the frozen configuration files under `../configs/`.
