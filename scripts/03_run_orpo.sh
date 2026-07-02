#!/usr/bin/env bash
# ORPO cold start across H200 GPUs (default 4 ranks).
# Usage: ./03_run_orpo.sh configs/orpo_lfm25_8b.env
set -euo pipefail

CONFIG="${1:-configs/orpo_lfm25_8b.env}"
if [[ ! -f "$CONFIG" ]]; then
  echo "config not found: $CONFIG" >&2
  exit 1
fi
# shellcheck disable=SC1090
source "$CONFIG"

# Load HF_TOKEN from ../../.env if not already set
if [[ -z "${HF_TOKEN:-}" ]]; then
  ENV_FILE="$(cd "$(dirname "$0")/../.." && pwd)/.env"
  if [[ -f "$ENV_FILE" ]]; then
    # shellcheck disable=SC1090
    source "$ENV_FILE"
  fi
fi

cd "$(dirname "$0")/.."

# Optional WANDB login (skip silently if not set)
if [[ -n "${WANDB_API_KEY:-}" ]]; then
  wandb login --relogin "$WANDB_API_KEY" >/dev/null 2>&1 || true
fi

export CUDA_VISIBLE_DEVICES="${TRAIN_GPUS}"
export HF_TOKEN
export WANDB_PROJECT="${WANDB_PROJECT:-graph-preflexor-lfm25}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-lfm25-orpo}"
export WANDB_NAME="${WANDB_NAME:-${WANDB_RUN_GROUP}-$(date +%Y%m%dT%H%M%SZ)}"

CMD=(
  torchrun --nproc_per_node="${NPROC_PER_NODE}" --master_port=29501
  -m src.orpo_train
  --base_model "${MODEL_ID}"
  --dataset_path "${PROCESSED_DATASET}"
  --output_dir "${OUTPUT_DIR}"
  --mode "${MODE}"
  --lora_target_modules "${LORA_TARGET_MODULES}"
  --lora_r "${LORA_R}"
  --lora_alpha "${LORA_ALPHA}"
  --lora_dropout "${LORA_DROPOUT}"
  --lr "${LR}"
  --epochs "${EPOCHS}"
  --batch_size "${BATCH_SIZE}"
  --grad_accum "${GRAD_ACCUM}"
  --max_length "${MAX_LENGTH}"
  --save_steps "${SAVE_STEPS}"
  --eval_steps "${EVAL_STEPS}"
  --logging_steps "${LOGGING_STEPS}"
  --chat_template_enable_thinking "${CHAT_TEMPLATE_ENABLE_THINKING}"
)

if [[ "${ADD_NEW_SPECIAL_TOKENS:-0}" == "1" ]]; then
  CMD+=(--add_new_special_tokens)
fi
if [[ "${HUB_PUSH:-1}" == "1" ]]; then
  CMD+=(--push_to_hub --hub_model_id "${HUB_MODEL_ID}")
  if [[ "${HUB_PUBLIC:-0}" == "1" ]]; then
    CMD+=(--hub_public)
  fi
fi

echo "[orpo] launching: ${CMD[*]}"
exec "${CMD[@]}"
