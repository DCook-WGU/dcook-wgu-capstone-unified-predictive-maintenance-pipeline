#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

echo "Rebuilding Postgres stack in: $(pwd)"

echo "Stopping existing containers..."
docker compose down

echo "Pulling latest images..."
docker compose pull

echo "Starting stack..."
docker compose up -d

echo "Postgres stack rebuilt and running."
