.PHONY: hooks
hooks:
	pre-commit install

.PHONY: test
test:
	uv run pytest

.PHONY: default-config
default-config:
	uv run --with inspect-ai,pyyaml --no-project tools/generate_default_config.py
