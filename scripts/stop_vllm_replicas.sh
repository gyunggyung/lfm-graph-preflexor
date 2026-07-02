#!/usr/bin/env bash
# Stop all vLLM replicas launched by run_vllm_replicas.sh
set -euo pipefail
pkill -f "vllm.entrypoints.openai.api_server" || true
echo "[vllm] stopped all api_server processes"
