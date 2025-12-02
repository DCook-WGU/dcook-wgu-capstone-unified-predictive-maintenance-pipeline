#!/bin/bash
#set -euo pipefail
set -eo pipefail

set +u

# Load Conda
if [ -f /opt/miniconda/etc/profile.d/conda.sh ]; then
    # shellcheck source=/dev/null
    source /opt/miniconda/etc/profile.d/conda.sh
else
    echo "[entrypoint] ERROR: conda.sh not found at /opt/miniconda/etc/profile.d/conda.sh" >&2
    exit 1
fi

# Default to "capstone" unless overridden at build/run time
: "${ENV_NAME:=capstone}"

# Add auto-activation to .bashrc if it's not already there
grep -qxF "conda activate $ENV_NAME" ~/.bashrc || echo "conda activate $ENV_NAME" >> ~/.bashrc

# Activate it now (for current session)
conda activate "$ENV_NAME"

set -u

# Ensure /app is on PYTHONPATH (for imports like src.*)
export PYTHONPATH="/app:${PYTHONPATH:-}"

# Optional: quick diagnostics
echo "[entrypoint] Conda env: ${ENV_NAME}"
echo "[entrypoint] PYTHONPATH: ${PYTHONPATH}"

# Log in to Weights & Biases (if both key and CLI are available)
if [ -n "${WANDB_API_KEY:-}" ] && command -v wandb >/dev/null 2>&1; then
    wandb login "$WANDB_API_KEY"
fi

# Launch interactive shell or passed command
exec "$@"
