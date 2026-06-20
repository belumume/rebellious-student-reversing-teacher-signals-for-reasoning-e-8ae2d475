#!/usr/bin/env bash
# Minimal end-to-end reproduction of the RLRT core mechanism (arXiv:2605.10781).
# Trains the SAME tiny model twice on a verifiable arithmetic-reasoning task --
# once with GRPO, once with RLRT (reversed teacher signal) -- and writes EVAL.md
# comparing them. Runs on a single GPU in minutes.
set -euo pipefail

export HF_HUB_ENABLE_HF_TRANSFER=1
export TOKENIZERS_PARALLELISM=false

PY=$(command -v python3 || command -v python)
echo "Using python: $PY"

echo "=== Installing dependencies ==="
$PY -m pip install -q --upgrade pip
$PY -m pip install -q "torch" "transformers>=4.44" "numpy" "accelerate" "hf_transfer" || \
  $PY -m pip install -q "transformers>=4.44" "numpy" "accelerate" "hf_transfer"

echo "=== GPU info ==="
$PY -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"

echo "=== Running GRPO vs RLRT ==="
cd rlrt_min
$PY train.py "$@"

echo "=== EVAL.md ==="
cat ../EVAL.md
