UV ?= uv

.PHONY: setup setup-evolution setup-dev show-config smoke test format check sandbox-image

setup:
	$(UV) sync --frozen --no-default-groups

setup-evolution:
	$(UV) sync --frozen --no-default-groups --extra evolution

setup-dev:
	$(UV) sync --frozen --group dev

show-config:
	$(UV) run prefix-cache-evolve --show-config

smoke:
	$(UV) run prefix-cache-evolve --baseline-report --quick

test:
	$(UV) run pytest -q

format:
	$(UV) run ruff format .

check:
	$(UV) run ruff format --check .
	$(UV) run ruff check .
	$(UV) run mypy
	$(UV) run pytest -q --cov=prefix_cache_evolve --cov-report=term-missing --cov-fail-under=75

sandbox-image:
	docker build --tag prefix-cache-evolve-sandbox --file docker/sandbox/Dockerfile .
