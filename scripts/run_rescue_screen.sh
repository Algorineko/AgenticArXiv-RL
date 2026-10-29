#!/usr/bin/env bash
# Fixed four-arm DEVELOPMENT screen. Each arm has a separate process and output.
set -euo pipefail
if [ "$#" -ne 4 ]; then
  echo "usage: $0 MODEL_ROOT CATALOG OUTPUT_ROOT CATALOG_MANIFEST_SHA256" >&2
  exit 2
fi
MODEL_ROOT="$1"
CATALOG="$2"
OUTPUT_ROOT="$3"
CATALOG_HASH="$4"
RESCUE_PYTHON="${RESCUE_PYTHON:-python3}"
RESCUE_GPU="${RESCUE_GPU:-1}"
mkdir -p "$OUTPUT_ROOT"
for arm in base_baseline grpo_baseline base_completion_check grpo_completion_check; do
  case "$arm" in
    base_*) model="$MODEL_ROOT/base-1.5b"; model_id="Qwen/Qwen2.5-1.5B-Instruct" ;;
    grpo_*) model="$MODEL_ROOT/grpo-1.5b"; model_id="Algorineko/AgenticArXiv-RL-Qwen2.5-1.5B-GRPO" ;;
  esac
  variant="${arm#*_}"
  echo "START $arm $(date -u +%FT%TZ)"
  "$RESCUE_PYTHON" scripts/rescue_replay_eval.py --catalog "$CATALOG" \
    --catalog-sha256 "$CATALOG_HASH" --split search_dev --policy model \
    --model "$model" --model-id "$model_id" --variant "$variant" \
    --gpu "$RESCUE_GPU" --timeout 900 --output "$OUTPUT_ROOT/$arm" \
    > "$OUTPUT_ROOT/$arm.console.log" 2>&1
  tail -n 1 "$OUTPUT_ROOT/$arm.console.log"
  echo "END $arm $(date -u +%FT%TZ)"
done
