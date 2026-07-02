#!/usr/bin/env bash
# Chained pipeline: ORPO -> GRPO -> eval-generate -> (judge) -> eval-collect -> upload
#
# Assumes ORPO is already running. This script polls for the ORPO checkpoint to
# appear, then starts GRPO, then eval, then upload. Each stage logs to logs/.
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG_ORPO="${1:-configs/orpo_qwen3_8b.env}"
CONFIG_GRPO="${2:-configs/grpo_qwen3_8b.env}"
MODEL_LABEL="${MODEL_LABEL:-Qwen3-8B-Graph-PRefLexOR-Repro}"
CHECKPOINT_LABEL="${CHECKPOINT_LABEL:-final}"

mkdir -p logs/pipeline

# Stage gate 1: wait for ORPO to finish (look for adapter_model.json)
source "$CONFIG_ORPO"
ORPO_OUT="${OUTPUT_DIR}"
echo "[pipeline] waiting for ORPO checkpoint at $ORPO_OUT/adapter_model.json"
until [ -f "$ORPO_OUT/adapter_model.json" ]; do
  sleep 60
done
# Additional: wait until no orpo_train process running
until ! pgrep -f "src.orpo_train" >/dev/null; do
  sleep 30
done
echo "[pipeline] ORPO done. Saved at $ORPO_OUT"

# Stage gate 2: run GRPO
source "$CONFIG_GRPO"
GRPO_OUT="${OUTPUT_DIR}"
echo "[pipeline] starting GRPO -> $GRPO_OUT"
HUB_PUSH=0 nohup bash scripts/04_run_grpo.sh "$CONFIG_GRPO" > logs/pipeline/grpo_run1.log 2>&1 &
GRPO_PID=$!
disown
echo "[pipeline] GRPO PID=$GRPO_PID, waiting..."
wait $GRPO_PID || true
echo "[pipeline] GRPO done"

if [ ! -f "$GRPO_OUT/adapter_model.json" ]; then
  echo "[pipeline] ERROR: GRPO did not produce adapter_model.json at $GRPO_OUT"
  exit 1
fi

# Stage gate 3: merge adapter into base for eval
MERGED_DIR="$GRPO_OUT/merged"
echo "[pipeline] merging adapter -> $MERGED_DIR"
"$PYTHON_BIN" - <<PYEOF
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import os, json
adapter_cfg = json.load(open("$GRPO_OUT/adapter_config.json"))
base = adapter_cfg["base_model_name_or_path"]
print(f"Loading base: {base}")
model = AutoModelForCausalLM.from_pretrained(base, dtype="bfloat16", device_map="cpu")
tok = AutoTokenizer.from_pretrained(base)
model = PeftModel.from_pretrained(model, "$GRPO_OUT")
model = model.merge_and_unload()
os.makedirs("$MERGED_DIR", exist_ok=True)
model.save_pretrained("$MERGED_DIR", safe_serialization=True)
tok.save_pretrained("$MERGED_DIR")
print(f"Merged saved to $MERGED_DIR")
PYEOF

# Stage gate 4: eval-generate
EVAL_LOG="logs/pipeline/eval_${MODEL_LABEL}.log"
echo "[pipeline] generating eval outputs -> $EVAL_LOG"
"$PYTHON_BIN" scripts/05_eval_benchmark.py \
  --model_path "$MERGED_DIR" \
  --model_label "$MODEL_LABEL" \
  --checkpoint_label "$CHECKPOINT_LABEL" \
  --generate \
  > "$EVAL_LOG" 2>&1

echo "[pipeline] eval generated. Now:"
echo "  bash scripts/run_judge_worker.sh  # Claude judges 3 metrics x 100 q"
echo "  Then rerun this script's collect+upload stage, or run:"
echo "  $PYTHON_BIN scripts/05_eval_benchmark.py --model_path $MERGED_DIR --model_label $MODEL_LABEL --collect"
echo "  $PYTHON_BIN scripts/06_upload_to_hub.py --local_dir $MERGED_DIR --hub_model_id <id> --base_model <base>"
