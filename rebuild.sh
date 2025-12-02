#!/usr/bin/env bash
set -euo pipefail

IMAGE="dcook_capstone_app:latest"

echo "[REBUILD] Stopping services that use $IMAGE…"
docker compose stop app db_consumer \
  producer_water_pump_normal producer_water_pump_mixed \
  producer_tep_normal producer_tep_mixed \
  producer_azure_pdm_normal producer_azure_pdm_mixed \
  producer_turbofan_normal producer_turbofan_mixed \
  producer_ai4i_normal producer_ai4i_mixed || true

echo "[REBUILD] Rebuilding application image: $IMAGE"
docker compose build app

echo "[REBUILD] Restarting updated services…"
docker compose up -d app db_consumer \
  producer_water_pump_normal producer_water_pump_mixed \
  producer_tep_normal producer_tep_mixed \
  producer_azure_pdm_normal producer_azure_pdm_mixed \
  producer_turbofan_normal producer_turbofan_mixed \
  producer_ai4i_normal producer_ai4i_mixed

echo "[REBUILD] Rebuild complete."
echo "Check logs with: docker compose logs -f app"
