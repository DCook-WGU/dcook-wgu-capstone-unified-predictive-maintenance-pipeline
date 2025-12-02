#!/bin/bash
set -euo pipefail

CONTAINER_NAME="${CONTAINER_NAME:-python_eda}"

echo "🛑 Stopping container: $CONTAINER_NAME (if running)"
docker stop "$CONTAINER_NAME" >/dev/null 2>&1 || true

echo "🧹 Removing container: $CONTAINER_NAME (if exists)"
docker rm "$CONTAINER_NAME" >/dev/null 2>&1 || true

# Optional GPU context cleanup
if command -v nvidia-smi &>/dev/null; then
  nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9 2>/dev/null || true
fi

echo "✅ Done. Container and GPU contexts cleaned up."


# Original version
#!/bin/bash
#docker stop mlops-container && docker rm mlops-container
