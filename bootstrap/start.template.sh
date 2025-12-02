#!/usr/bin/env bash
set -euo pipefail

# Template start script for any docker-compose stack.
# Copy this script into the stack folder and rename it to start.sh

cd "$(dirname "${BASH_SOURCE[0]}")"

echo "Starting stack in: $(pwd)"
docker compose up -d

echo "Stack started."
