FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

RUN mkdir -p /app/logs

RUN mkdir -p /app/artifacts

ENV CONFIG=./config/model1.yaml

COPY . /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/* # Clean up apt cache to reduce image size

# Use uv instead of pip for faster installations and improved dependency resolution.
RUN uv pip install -e . --system

# Install SHAP directly from the GitHub repository because the version available on PyPI wasn't compatible with XGBoost 3.2.0
RUN uv pip install git+https://github.com/shap/shap.git --system