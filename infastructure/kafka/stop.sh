#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

echo "Stopping Kafka stack in: $(pwd)"
docker compose down

echo "Kafka stack stopped."
