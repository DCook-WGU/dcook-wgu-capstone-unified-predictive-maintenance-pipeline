#!/usr/bin/env bash
set -euo pipefail

# Template stop script for any docker-compose stack.
# Copy this script into the stack folder and rename it to stop.sh

cd "$(dirname "${BASH_SOURCE[0]}")"

echo "Stopping stack in: $(pwd)"
docker compose down

echo "Stack stopped."
