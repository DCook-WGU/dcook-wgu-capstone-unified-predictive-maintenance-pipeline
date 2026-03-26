#!/usr/bin/env bash
set -euo pipefail

# Always run docker compose commands from the folder this script is in
cd "$(dirname "${BASH_SOURCE[0]}")"

echo "Stopping Postgres stack in: $(pwd)"
docker compose down

echo "Postgres stack stopped."
