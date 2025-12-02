#!/usr/bin/env bash
set -euo pipefail

# Template rebuild script for any docker-compose stack.
# Copy this script into the stack folder and rename it to rebuild.sh

cd "$(dirname "${BASH_SOURCE[0]}")"

echo "Rebuilding stack in: $(pwd)"

echo "Stopping containers..."
docker compose down

echo "Pulling latest images..."
docker compose pull

echo "Starting containers..."
docker compose up -d

echo "Stack rebuilt and running."
