#!/bin/bash

# Make script strict. Will not fail silently. 
set -euo pipefail

# Defaults you can override per folder or at runtime:
#   CONTAINER_NAME=mlops-project ./start.sh
CONTAINER_NAME="${CONTAINER_NAME:-python_eda}"
IMAGE="${IMAGE:-python_eda}"


# Check that .env exists and has WANDB_API_KEY defined.
# Warn (but don't fail) if WANDB_API_KEY isn't set in your WSL env
if [[ ! -f ".env" ]]; then
  echo "[WARN] .env file not found in project root."
  echo "       W&B variables (WANDB_API_KEY / PROJECT / ENTITY) will not be loaded into containers."
else
  if ! grep -q '^WANDB_API_KEY=' .env; then
    echo "[WARN] WANDB_API_KEY is not set in .env."
    echo "       W&B runs may not be authenticated."
  fi
fi

echo "Starting container: $CONTAINER_NAME (image: $IMAGE)"
docker run -it \
  --gpus all \
  --name "$CONTAINER_NAME" \
  --rm \
  -v "$(pwd)":/app \
  -e WANDB_API_KEY="${WANDB_API_KEY:-}" \
  "$IMAGE"


# Original build
#!/bin/bash

# Optional: Export your API key once per shell session
#export WANDB_API_KEY=your-api-key-here

#docker run -it \
#  --name mlops-container \
#  -v $(pwd):/app \
#  -e WANDB_API_KEY=$WANDB_API_KEY \
#  mlops-dev






