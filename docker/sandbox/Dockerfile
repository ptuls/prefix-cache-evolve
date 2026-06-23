FROM ghcr.io/astral-sh/uv:0.9.18 AS uv

FROM python:3.11-slim

COPY --from=uv /uv /usr/local/bin/uv

RUN useradd --create-home --uid 10001 evaluator

WORKDIR /opt/prefix-cache-evolve
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY configs ./configs
COPY src ./src
COPY docker/sandbox/entrypoint.sh /usr/local/bin/prefix-cache-sandbox

RUN uv sync --frozen --no-default-groups --no-editable \
    && chmod 0555 /usr/local/bin/prefix-cache-sandbox

ENV PATH="/opt/prefix-cache-evolve/.venv/bin:${PATH}"

USER evaluator
WORKDIR /tmp

ENTRYPOINT ["/usr/local/bin/prefix-cache-sandbox"]
