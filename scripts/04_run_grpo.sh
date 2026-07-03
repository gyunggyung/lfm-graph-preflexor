#!/usr/bin/env bash
# Graph-GRPO with embedding reward + Claude judge queue.
#
# Pre-reqs:
#   1. vLLM rollout servers (one per VLLM_GPUS) already running, serving the
#      merged ORPO model on each VLLM_BASE_PORT + i. Use scripts/run_vllm_replicas.sh.
#   2. ./checkpoints/orpo_<model> exists (or BASE_MODEL_DIR points to a Hub adapter)
#
# Usage: ./04_run_grpo.sh configs/grpo_qwen3_8b.env
set -euo pipefail

CONFIG="${1:-configs/grpo_qwen3_8b.env}"
if [[ ! -f "$CONFIG" ]]; then
  echo "config not found: $CONFIG" >&2
  exit 1
fi
# shellcheck disable=SC1090
source "$CONFIG"

if [[ -z "${HF_TOKEN:-}" ]]; then
  ENV_FILE="$(cd "$(dirname "$0")/../.." && pwd)/.env"
  if [[ -f "$ENV_FILE" ]]; then
    HF_TOKEN="$(grep -E "^export HF_TOKEN=" "$ENV_FILE" | sed 's/^export HF_TOKEN=//' | tr -d '"' | tr -d "'")"
    export HF_TOKEN
  fi
fi

cd "$(dirname "$0")/.."

if [[ -n "${WANDB_API_KEY:-}" ]]; then
  "$PYTHON_BIN" -m wandb login --relogin "$WANDB_API_KEY" >/dev/null 2>&1 || true
fi

export CUDA_VISIBLE_DEVICES="${TRAIN_GPUS}"
export HF_TOKEN
export LD_LIBRARY_PATH="${VLLM_LD_LIBRARY_PATH}"
export PYTHONPATH="./:${PYTHONPATH:-}"
export WANDB_NAME="${WANDB_NAME:-${WANDB_RUN_GROUP}-$(date +%Y%m%dT%H%M%SZ)}"

# Pick a vLLM endpoint: by default, first port (VLLM_BASE_PORT).
FIRST_PORT="${VLLM_BASE_PORT%%,*}"
VLLM_BASE_URL="http://${VLLM_HOST:-127.0.0.1}:${FIRST_PORT}/v1"

if [[ "${GRPO_NO_VLLM:-0}" != "1" && "${VLLM_SKIP_CHECK:-0}" != "1" ]]; then
  if curl -fsS "${VLLM_BASE_URL}/models" >/dev/null 2>&1; then
    echo "[grpo] vLLM endpoint OK: ${VLLM_BASE_URL}"
  else
    echo "[grpo] WARNING: vLLM endpoint not responding at ${VLLM_BASE_URL}" >&2
    echo "[grpo]          start it with scripts/run_vllm_replicas.sh or set VLLM_SKIP_CHECK=1" >&2
    exit 1
  fi
fi

MASTER_PORT="${MASTER_PORT:-29502}"

# vLLM colocate (default) vs server mode
VLLM_MODE_EFF="${VLLM_MODE:-colocate}"

CMD=(
  env -u PYTHONNOUSERSITE
  "$PYTHON_BIN" -m torch.distributed.run
  --nproc_per_node="${NPROC_PER_NODE}" --master_port="${MASTER_PORT}"
  -m src.grpo_train
  --base_model_dir "${BASE_MODEL_DIR}"
  --tokenizer_model "${TOKENIZER_MODEL}"
  --dataset_path "${PROCESSED_DATASET}"
  --output_dir "${OUTPUT_DIR}"
  --model_name_label "${WANDB_RUN_GROUP}"
  --lora_target_modules "${LORA_TARGET_MODULES}"
  --lora_r "${LORA_R}"
  --lora_alpha "${LORA_ALPHA}"
  --lora_dropout "${LORA_DROPOUT}"
  --per_device_train_batch_size "${PER_DEVICE_BATCH}"
  --gradient_accumulation_steps "${GRAD_ACCUM}"
  --num_generations "${NUM_GENERATIONS}"
  --learning_rate "${LR}"
  --epochs "${EPOCHS}"
  --max_prompt_length "${MAX_PROMPT_LENGTH}"
  --max_completion_length "${MAX_COMPLETION_LENGTH}"
  --temperature "${TEMPERATURE}"
  --save_steps "${SAVE_STEPS}"
  --logging_steps "${LOGGING_STEPS}"
  --scale_rewards "${SCALE_REWARDS}"
  --loss_type "${LOSS_TYPE}"
  --weight_correctness "${WEIGHT_CORRECTNESS}"
  --weight_format "${WEIGHT_FORMAT}"
  --weight_graph_utility "${WEIGHT_GRAPH_UTILITY}"
  --weight_graph_networkx "${WEIGHT_GRAPH_NETWORKX}"
  --weight_graph_diversity "${WEIGHT_GRAPH_DIVERSITY}"
  --weight_graph_structure "${WEIGHT_GRAPH_STRUCTURE}"
  --embed_model "${EMBED_MODEL}"
  --judge_queue_dir "${JUDGE_QUEUE_DIR}"
  --judge_queue_every_steps "${JUDGE_QUEUE_EVERY_STEPS}"
  --judge_queue_batch_size "${JUDGE_QUEUE_BATCH_SIZE}"
  --chat_template_enable_thinking "${CHAT_TEMPLATE_ENABLE_THINKING}"
)
if [[ "${GRPO_NO_VLLM:-0}" != "1" ]]; then
  CMD+=(--use_vllm --vllm_mode "${VLLM_MODE_EFF}")
  if [[ "${VLLM_MODE_EFF}" == "server" ]]; then
    CMD+=(--vllm_server_host "${VLLM_HOST:-127.0.0.1}" --vllm_server_port "${FIRST_PORT}")
  else
    CMD+=(--vllm_gpu_memory_utilization "${VLLM_COLOCATE_GPU_MEM_UTIL:-0.35}")
  fi
fi

if [[ -n "${CLAUDE_BLEND_ALPHA:-}" ]]; then
  CMD+=(--claude_blend_alpha "${CLAUDE_BLEND_ALPHA}")
fi
if [[ "${ADD_NEW_SPECIAL_TOKENS:-0}" == "1" ]]; then
  CMD+=(--add_new_special_tokens)
fi
if [[ -n "${RESUME_GRPO_CHECKPOINT:-}" ]]; then
  CMD+=(--resume_grpo_checkpoint "${RESUME_GRPO_CHECKPOINT}")
fi
if [[ "${HUB_PUSH:-0}" == "1" ]]; then
  CMD+=(--push_to_hub --hub_model_id "${HUB_MODEL_ID}")
  if [[ "${HUB_PUBLIC:-0}" == "1" ]]; then
    CMD+=(--hub_public)
  fi
fi

echo "[grpo] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} NPROC=${NPROC_PER_NODE} PYTHON=${PYTHON_BIN}"
echo "[grpo] launching: ${CMD[*]}"
exec "${CMD[@]}"
