#!/usr/bin/env bash
# Serve gpt-oss-120b (openai/gpt-oss-120b) with vLLM 0.29: OpenAI-compatible API on port 8103.
# The openai_gptoss parser returns the reasoning text separately from the reply, and the
# openai tool-call parser provides native tool calling.
#   MODEL_DIR=/path/to/gpt-oss-120b harness/serving/serve_gpt_oss.sh
set -euo pipefail
exec vllm serve "${MODEL_DIR:?set MODEL_DIR to the directory of the model weights}" \
  --served-model-name gpt-oss-120b \
  --host 127.0.0.1 --port 8103 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 131072 \
  --reasoning-parser openai_gptoss \
  --enable-auto-tool-choice --tool-call-parser openai \
  --max-num-seqs 64 --seed 0
