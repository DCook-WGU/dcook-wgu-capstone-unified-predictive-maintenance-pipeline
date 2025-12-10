#!/usr/bin/env bash

# Make script strict. Will not fail silently.
set -euo pipefail

echo "[START] Bringing up entire capstone pipeline…"

# Check that .env exists and has WANDB_API_KEY defined.
# Warn (but don't fail) if WANDB_API_KEY isn't set
if [[ ! -f ".env" ]]; then
  echo "[WARN] .env file not found in project root."
  echo "       W&B variables (WANDB_API_KEY / PROJECT / ENTITY) will not be loaded into containers."
else
  if ! grep -q '^WANDB_API_KEY=' .env; then
    echo "[WARN] WANDB_API_KEY is not set in .env."
    echo "       W&B runs may not be authenticated."
  fi
fi


docker compose up -d

echo "[START] All services launched."

echo "Run: docker compose logs -f app   (or any service name)"