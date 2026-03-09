FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

WORKDIR /app

RUN mkdir -p /app/logs

# Project metadata first for caching
COPY pyproject.toml setup.py /app/
RUN uv pip install -e . --system

COPY . /app

ENV CONFIG=/app/config/model1.yaml
ENV PYTHONPATH=/app

CMD ["python", "-m", "pytest", "tests"]