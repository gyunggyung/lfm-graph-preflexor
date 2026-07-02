#!/usr/bin/env bash
# Launch vLLM rollout servers serving the merged ORPO model on the GPU list
# specified by VLLM_GPUS, one process per GPU. Use this BEFORE 04_run_grpo.sh.
#
# Usage: ./run_vllm_replicas.sh configs/grpo_lfm25_8b.env
set -euo pipefail

CONFIG="${1:-configs/grpo_lfm25_8b.env}"
# shellcheck disable=SC1090
source "$CONFIG"

if [[ -z "${HF_TOKEN:-}" ]]; then
  ENV_FILE="$(cd "$(dirname "$0")/../.." && pwd)/.env"
  if [[ -f "$ENV_FILE" ]]; then
    # shellcheck disable=SC1090
    source "$ENV_FILE"
  fi
fi

cd "$(dirname "$0")/.."

IFS=',' read -ra GPU_ARR <<< "${VLLM_GPUS}"
IFS=',' read -ra PORT_ARR <<< "${VLLM_PORTS}"

if [[ ${#GPU_ARR[@]} -ne ${#PORT_ARR[@]} ]]; then
  echo "VLLM_GPUS and VLLM_PORTS length mismatch" >&2
  exit 1
fi

LOG_DIR="./logs/vllm"
mkdir -p "$LOG_DIR"

SERVED="${VLLM_SERVED_MODEL:-${BASE_MODEL_DIR}}"

for i in "${!GPU_ARR[@]}"; do
  gpu="${GPU_ARR[$i]}"
  port="${PORT_ARR[$i]}"
  log="${LOG_DIR}/vllm_gpu${gpu}_port${port}.log"
  echo "[vllm] GPU ${gpu} -> port ${port} -> log ${log}"
  CUDA_VISIBLE_DEVICES="${gpu}" nohup python -m vllm.entrypoints.openai.api_server \
    --model "${SERVED}" \
    --served-model-name "graph-preflexor-grpo" \
    --port "${port}" \
    --host 127.0.0.1 \
    --gpu-memory-utilization "${VLLM_GPU_MEM_UTIL:-0.85}" \
    --max-model-len "${VLLM_MAX_MODEL_LEN:-8192}" \
    --trust-remote-code \
    >"${log}" 2>&1 &
  echo "[vllm] launched pid=$!"
done

echo "[vllm] all replicas launched. Wait for 'Application startup complete' in logs."
echo "[vllm] health check:"
for port in "${PORT_ARR[@]}"; do
  echo "  curl -fsS http://127.0.0.1:${port}/v1/models"
done
