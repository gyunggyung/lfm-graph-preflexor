# Common environment for graph-preflexor-lfm25 training.
# Sourced by all 03_run_*.sh and 04_run_*.sh launchers.
# Override per-model in the model-specific config.

# ---- GPU layout: 4 GPUs (2,3,4,5) ----
# Train uses all 4 with DDP; vLLM runs colocate on the same GPUs.
export TRAIN_GPUS="${TRAIN_GPUS:-2,3,4,5}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"

# ---- Python env (use existing vllm env so torch/trl/vllm versions match) ----
export TERMINAL_ROOT="${TERMINAL_ROOT:-/home/work/.projects/LLM-OS-Models/Terminal}"
# Use the strict cu129 env: vllm 0.20.2 works with Qwen3.5 text-only arch
# (.vllm-lfm-cu12 has vllm 0.19.1 which fails on Qwen3_5TextForCausalLM)
export VLLM_ENV="${VLLM_ENV:-$TERMINAL_ROOT/.vllm-eval-cu129-strict}"
export PYTHON_BIN="${PYTHON_BIN:-$VLLM_ENV/bin/python}"

# LD_LIBRARY_PATH for torch/cuda/nvidia libs in vllm-env
_VENV_SITE="$VLLM_ENV/lib/python3.12/site-packages"
export VLLM_LD_LIBRARY_PATH="${VLLM_LD_LIBRARY_PATH:-$_VENV_SITE/torch/lib:$_VENV_SITE/nvidia/cuda_runtime/lib:$_VENV_SITE/nvidia/cu13/lib:$_VENV_SITE/nvidia/cublas/lib:$_VENV_SITE/nvidia/cudnn/lib:$_VENV_SITE/nvidia/nccl/lib:$_VENV_SITE/nvidia/cusparselt/lib:$_VENV_SITE/nvidia/nvshmem/lib:/usr/local/cuda/compat/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}}"

# ---- vLLM colocate (default — no external server needed) ----
export VLLM_MODE="${VLLM_MODE:-colocate}"
export VLLM_COLOCATE_GPU_MEM_UTIL="${VLLM_COLOCATE_GPU_MEM_UTIL:-0.35}"

# ---- External vLLM server (only used if VLLM_MODE=server) ----
export VLLM_BASE_PORT="${VLLM_BASE_PORT:-8123}"
export VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
export VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-8192}"
export VLLM_GPU_MEM_UTIL="${VLLM_GPU_MEM_UTIL:-0.85}"
export VLLM_SERVED_NAME="${VLLM_SERVED_NAME:-graph-preflexor-grpo}"
export VLLM_LOG_DIR="${VLLM_LOG_DIR:-./logs/vllm}"

# ---- HuggingFace ----
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"

# ---- Misc ----
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export WANDB_PROJECT="${WANDB_PROJECT:-graph-preflexor-lfm25}"
