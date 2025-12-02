#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <container_name>"
  exit 1
fi

docker exec -it "$1" bash
