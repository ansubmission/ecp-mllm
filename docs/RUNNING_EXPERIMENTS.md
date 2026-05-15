# Running experiments

Representative entry points:

- sonar direct baseline: `python -m ecp_mllm.experiments.prompt_batch ...`
- sonar iterative inference: `python -m ecp_mllm.experiments.event_agent_batch ...`
- thermal direct baseline: `python -m ecp_mllm.experiments.thermal_prompt_batch ...`
- thermal iterative inference: `python -m ecp_mllm.experiments.thermal_event_agent_batch ...`
- compact adaptation training: `python -m ecp_mllm.experiments.train_multimodal_qlora ...`
- compact adaptation sweep: `python -m ecp_mllm.experiments.run_event_harness ...`

Representative benchmark configs retained in this package:

- `config/prompt_dev/kenai_val_holdout_8clips_v1.json`
- `config/prompt_dev/elwha_holdout_8clips_v1.json`
- `config/prompt_dev/elwha_high_holdout_8clips_v1.json`
- `config/prompt_dev/nz_thermal_mixed8_rescue_v1.json`
- `config/prompt_dev/nz_thermal_zone6_meaningful_v1.json`
- `config/prompt_dev/cfc_capability_probe8_v1.json`
- `config/prompt_dev/thermal_capability_probe8_v1.json`

Local paths, model access, and benchmark roots are configured through
`config/local.example.toml`.
