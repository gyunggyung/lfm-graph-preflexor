#!/usr/bin/env bash
# Launch one vLLM api_server per GPU in VLLM_GPUS.
#
# Uses configs/env_common.sh for VLLM_ENV, LD_LIBRARY_PATH, ports.
# Adapted from Liquid-CLI/scripts/run_lfm25_vllm_replicas_clean.sh.
#
# Usage:
#   ./run_vllm_replicas.sh configs/grpo_qwen3_8b.env [model_path]
#
# If model_path is not given, defaults to BASE_MODEL_DIR/merged from config,
# falling back to TOKENIZER_MODEL (base HF model).
set -euo pipefail

CONFIG="${1:-configs/grpo_qwen3_8b.env}"
if [[ ! -f "$CONFIG" ]]; then
  echo "config not found: $CONFIG" >&2
  exit 1
fi
# shellcheck disable=SC1090
source "$CONFIG"

MODEL_PATH="${2:-${BASE_MODEL_DIR}}"
# Backward compat: if BASE_MODEL_DIR has no config.json but BASE_MODEL_DIR/merged does, use that
if [[ ! -f "$MODEL_PATH/config.json" && -f "$MODEL_PATH/merged/config.json" ]]; then
  MODEL_PATH="$MODEL_PATH/merged"
fi
if [[ ! -f "$MODEL_PATH/config.json" ]]; then
  echo "[vllm] no config.json at $MODEL_PATH, falling back to ${TOKENIZER_MODEL}"
  MODEL_PATH="${TOKENIZER_MODEL}"
fi

mkdir -p "$VLLM_LOG_DIR"
rm -f "$VLLM_LOG_DIR/pids.txt" "$VLLM_LOG_DIR/urls.txt" "$VLLM_LOG_DIR/vllm_base_urls.txt"

IFS=',' read -r -a GPUS <<< "$VLLM_GPUS"
PIDS=()
URLS=()

cleanup() {
  echo "[vllm] cleaning up replicas..."
  for pid in "${PIDS[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
}
trap cleanup INT TERM
READY_TIMEOUT_SEC="${VLLM_READY_TIMEOUT:-300}"

for i in "${!GPUS[@]}"; do
  gpu="${GPUS[$i]}"
  port="$((VLLM_BASE_PORT + i))"
  url="http://${VLLM_HOST}:${port}/v1"
  URLS+=("$url")

  echo "[vllm] starting gpu=${gpu} port=${port} model=${MODEL_PATH}"
  env -u PYTHONPATH -u PYTHONNOUSERSITE \
    LD_LIBRARY_PATH="$VLLM_LD_LIBRARY_PATH" \
    CUDA_VISIBLE_DEVICES="$gpu" \
    "$PYTHON_BIN" -m vllm.entrypoints.openai.api_server \
      --host "$VLLM_HOST" \
      --port "$port" \
      --model "$MODEL_PATH" \
      --served-model-name "$VLLM_SERVED_NAME" \
      --trust-remote-code \
      --dtype bfloat16 \
      --max-model-len "$VLLM_MAX_MODEL_LEN" \
      --gpu-memory-utilization "$VLLM_GPU_MEM_UTIL" \
      --tensor-parallel-size 1 \
      --enforce-eager \
      > "$VLLM_LOG_DIR/vllm_gpu${gpu}_port${port}.log" 2>&1 &
  PIDS+=("$!")

  ready=0
  for _attempt in $(seq 1 "$READY_TIMEOUT_SEC"); do
    if ! kill -0 "${PIDS[-1]}" 2>/dev/null; then
      echo "[vllm] replica exited before ready gpu=${gpu} port=${port}" >&2
      tail -n 100 "$VLLM_LOG_DIR/vllm_gpu${gpu}_port${port}.log" >&2 || true
      exit 1
    fi
    if curl -fsS "$url/models" >/dev/null 2>&1; then
      echo "[vllm] ready gpu=${gpu} url=${url}"
      ready=1
      break
    fi
    sleep 1
  done
  if [[ "$ready" != "1" ]]; then
    echo "[vllm] did not become ready in ${READY_TIMEOUT_SEC}s gpu=${gpu}" >&2
    tail -n 100 "$VLLM_LOG_DIR/vllm_gpu${gpu}_port${port}.log" >&2 || true
    exit 1
  fi
done

printf '%s\n' "${PIDS[@]}" > "$VLLM_LOG_DIR/pids.txt"
printf '%s\n' "${URLS[@]}" > "$VLLM_LOG_DIR/urls.txt"
COMMA_URLS="$(paste -sd, "$VLLM_LOG_DIR/urls.txt")"
printf '%s\n' "$COMMA_URLS" > "$VLLM_LOG_DIR/vllm_base_urls.txt"
echo "[vllm] all replicas up. urls=${COMMA_URLS}"
echo "[vllm] pids saved to $VLLM_LOG_DIR/pids.txt"
echo "[vllm] logs at $VLLM_LOG_DIR/vllm_gpu*_port*.log"

wait
