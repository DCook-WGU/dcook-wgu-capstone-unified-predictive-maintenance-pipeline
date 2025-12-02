#!/usr/bin/env bash
set -euo pipefail

echo "[STOP] Shutting down capstone pipeline…"

docker compose down

echo "[STOP] All services stopped and removed."
