#!/usr/bin/env bash
set -euo pipefail

# ------------------------------
# logs.sh — view logs for any service in docker-compose.yml
#
# Usage:
#   ./logs.sh app
#   ./logs.sh kafka
#   ./logs.sh db_consumer
#   ./logs.sh app --no-follow
# ------------------------------

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <service> [--no-follow]"
    echo "Example: $0 app"
    exit 1
fi

SERVICE="$1"
FOLLOW="1"

# Optional flag: --no-follow
if [[ "${2:-}" == "--no-follow" ]]; then
    FOLLOW="0"
fi

# Check if service exists in docker compose
if ! docker compose config --services | grep -qw "$SERVICE"; then
    echo "Error: service '$SERVICE' not found in docker-compose.yml."
    echo "Available services:"
    docker compose config --services
    exit 1
fi

echo "Showing logs for service: $SERVICE"

if [[ "$FOLLOW" == "1" ]]; then
    docker compose logs -f "$SERVICE"
else
    docker compose logs "$SERVICE"
fi
