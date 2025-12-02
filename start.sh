#!/usr/bin/env bash

# Make script strict. Will not fail silently.
set -euo pipefail

echo "[START] Bringing up entire capstone pipeline…"

# Warn (but don't fail) if WANDB_API_KEY isn't set
if [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "WANDB_API_KEY is not set. Set it in ~/.bashrc or export it before running."
fi

docker compose up -d

echo "[START] All services launched."

echo "Run: docker compose logs -f app   (or any service name)"