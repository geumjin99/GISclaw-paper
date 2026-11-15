#!/usr/bin/env bash
# Serve Qwen3.8-27B (Qwen/Qwen3.8-27B-FP8) with vLLM 0.29: OpenAI-compatible API on port 8102.
# Thinking is on by the model's default; the qwen3 parser splits the reasoning text off the
# reply. Sampling defaults come from the checkpoint's generation_config.json; the harness
# sends and records the per-request settings of harness/config.yaml. The qwen3_coder
# tool-call parser provides native tool calling.
#   MODEL_DIR=/path/to/Qwen3.8-27B-FP8 harness/serving/serve_qwen.sh
set -euo pipefail
exec vllm serve "${MODEL_DIR:?set MODEL_DIR to the directory of the model weights}" \
  --served-model-name qwen3.8-27b-fp8 \
  --host 127.0.0.1 --port 8102 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 131072 \
  --reasoning-parser qwen3 \
  --limit-mm-per-prompt '{"image": 0, "video": 0}' \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --max-num-seqs 64 --seed 0
