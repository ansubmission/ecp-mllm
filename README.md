# ECP-MLLM

This package contains the core code paths and benchmark configurations used for
event-centric experiments on weak-signal physical streams. It covers frontier inference,
compact-model evaluation, lightweight adaptation, and the supporting sonar and
thermal evaluation utilities.

The repository is organized as follows:

- `src/ecp_mllm/`: core library and experiment entry points.
- `config/`: benchmark configs for sonar and thermal slices.
- `runtime/`: training and evaluation wrappers for compact local runs.
- `local_host/`: optional local serving utilities for compact models.
- `tests/`: a focused regression suite for the core logic.
- `docs/`: package notes, experiment scope, and execution references.

Recommended starting points:

- `docs/CODE_ORIENTATION.md`
- `docs/RUNNING_EXPERIMENTS.md`
- `docs/EXPERIMENT_SCOPE.md`
