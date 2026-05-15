# Code orientation

## Event-centric perception in physical streams

- structured event representation: `src/ecp_mllm/types.py`
- event-centric task logic: `src/ecp_mllm/agent/event_centric.py`
- counting and thermal metrics: `src/ecp_mllm/eval/counting.py`, `src/ecp_mllm/eval/thermal_eval.py`

## Frontier inference

- direct prompting: `src/ecp_mllm/experiments/prompt_batch.py`, `src/ecp_mllm/experiments/thermal_prompt_batch.py`
- iterative event-centric inference: `src/ecp_mllm/experiments/event_agent_batch.py`, `src/ecp_mllm/experiments/thermal_event_agent_batch.py`
- repeated evaluation: `src/ecp_mllm/experiments/repeated_event_agent_batch.py`, `src/ecp_mllm/experiments/repeated_thermal_event_agent_batch.py`

## Compact adaptation

- local trainer: `src/ecp_mllm/experiments/train_multimodal_qlora.py`
- harness: `src/ecp_mllm/experiments/run_event_harness.py`
- local evaluation: `src/ecp_mllm/experiments/eval_event_window_batch.py`
- runtime wrappers: `runtime/scripts/train_qwen9b_qlora.sh`, `runtime/scripts/run_event_lora_6h_harness.sh`
