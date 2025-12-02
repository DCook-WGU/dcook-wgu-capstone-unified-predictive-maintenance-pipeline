# Makefile for nuclear_powerplant_sensors

IMAGE ?= capstone
CONTAINER ?= capstone
NO_CACHE ?= 0

.PHONY: help rebuild run stop rc exec logs prune gpu-check

help:
	@echo "Targets:"
	@echo "  make rebuild       - Rebuild Docker image (uses NO_CACHE=$(NO_CACHE))"
	@echo "  make run           - Start interactive container"
	@echo "  make stop          - Stop & remove container"
	@echo "  make rc            - Rebuild then auto-commit updated lockfiles"
	@echo "  make exec          - Exec into running container (bash)"
	@echo "  make logs          - Tail container logs"
	@echo "  make prune         - Prune dangling Docker resources"
	@echo "  make gpu-check     - Run torch.cuda check inside container"

rebuild:
	@echo "Rebuilding image: $(IMAGE) (NO_CACHE=$(NO_CACHE))"
	@NO_CACHE=$(NO_CACHE) ./rebuild.sh

run:
	@echo "Starting interactive container: $(CONTAINER)"
	@CONTAINER_NAME=$(CONTAINER) IMAGE=$(IMAGE) ./start.sh

stop:
	@echo "Stopping & removing: $(CONTAINER)"
	@CONTAINER_NAME=$(CONTAINER) ./stop.sh

# Rebuild + auto-commit lockfiles exported during Docker build
rc: rebuild
	@echo "Committing lockfiles (if changed)…"
	@git add environment.lock.yml requirements.txt 2>/dev/null || true
	@git commit -m "chore(env): update lockfiles [auto]" || echo "✅ No lockfile changes."

exec:
	@echo "Exec into: $(CONTAINER)"
	@docker exec -it $(CONTAINER) bash

logs:
	@echo "Logs: $(CONTAINER) (Ctrl+C to stop)"
	@docker logs -f $(CONTAINER)

prune:
	@echo "Pruning dangling Docker resources…"
	@docker system prune -f --volumes

# 🔍 GPU verification
gpu-check:
	@echo "Checking GPU availability inside container..."
	@docker exec -it $(CONTAINER) python - <<'PY'
import torch
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device:", torch.cuda.get_device_name(0))
else:
    print("No GPU detected inside container.")
PY
