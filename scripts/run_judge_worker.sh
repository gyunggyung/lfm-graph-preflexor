#!/usr/bin/env bash
# Convenience wrapper to invoke the judge worker. Run this when Claude is invoked.
#
# Usage:
#   ./run_judge_worker.sh --list-only          # check queue depth
#   ./run_judge_worker.sh                       # process all batches
#   ./run_judge_worker.sh --batch 20260702T..   # process one batch
set -euo pipefail
cd "$(dirname "$0")/.."
exec python scripts/judge_worker.py "$@"
