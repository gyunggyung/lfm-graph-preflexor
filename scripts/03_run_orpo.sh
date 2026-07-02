#!/usr/bin/env bash
# ORPO cold start using vllm-env python (so torch/trl/peft versions match H200).
# Usage: ./03_run_orpo.sh configs/orpo_qwen3_8b.env
set -euo pipefail

CONFIG="${1:-configs/orpo_qwen3_8b.env}"
if [[ ! -f "$CONFIG" ]]; then
  echo "config not found: $CONFIG" >&2
  exit 1
fi
# shellcheck disable=SC1090
source "$CONFIG"

# Load HF_TOKEN from ../../.env if not already set (without sourcing — the .env
# contains a `huggingface-cli login` line that breaks sourcing).
if [[ -z "${HF_TOKEN:-}" ]]; then
  ENV_FILE="$(cd "$(dirname "$0")/../.." && pwd)/.env"
  if [[ -f "$ENV_FILE" ]]; then
    HF_TOKEN="$(grep -E "^export HF_TOKEN=" "$ENV_FILE" | sed 's/^export HF_TOKEN=//' | tr -d '"' | tr -d "'")"
    export HF_TOKEN
  fi
fi

cd "$(dirname "$0")/.."

# Optional WANDB login (skip silently if not set)
if [[ -n "${WANDB_API_KEY:-}" ]]; then
  "$PYTHON_BIN" -m wandb login --relogin "$WANDB_API_KEY" >/dev/null 2>&1 || true
fi

export CUDA_VISIBLE_DEVICES="${TRAIN_GPUS}"
export HF_TOKEN
export LD_LIBRARY_PATH="${VLLM_LD_LIBRARY_PATH}"
export PYTHONPATH="./:${PYTHONPATH:-}"

# Resolve TRAIN_GPUS list -> CUDA_VISIBLE_DEVICES already does the remapping,
# so torchrun --rdzv_endpoint just needs a port.
MASTER_PORT="${MASTER_PORT:-29501}"

CMD=(
  env -u PYTHONNOUSERSITE
  "$PYTHON_BIN" -m torch.distributed.run
  --nproc_per_node="${NPROC_PER_NODE}" --master_port="${MASTER_PORT}"
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
if [[ "${HUB_PUSH:-0}" == "1" ]]; then
  CMD+=(--push_to_hub --hub_model_id "${HUB_MODEL_ID}")
  if [[ "${HUB_PUBLIC:-0}" == "1" ]]; then
    CMD+=(--hub_public)
  fi
fi

echo "[orpo] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} NPROC=${NPROC_PER_NODE} PYTHON=${PYTHON_BIN}"
echo "[orpo] launching: ${CMD[*]}"
exec "${CMD[@]}"
