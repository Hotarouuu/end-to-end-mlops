FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

WORKDIR /app

RUN mkdir -p /app/logs

ENV CONFIG=./config/model1.yaml

COPY . /app

# Use uv instead of pip for faster installations and improved dependency resolution.
RUN uv pip install -e . --system