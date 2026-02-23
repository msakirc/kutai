FROM python:3.12-slim

# Install system deps (git for the workspace, plus common build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install common Python packages the agent might need
RUN pip install --no-cache-dir \
    flask \
    fastapi \
    uvicorn[standard] \
    requests \
    httpx \
    pandas \
    numpy \
    pydantic \
    sqlalchemy \
    aiosqlite \
    pytest \
    black \
    pylint \
    ruff \
    click \
    rich \
    python-dotenv \
    jinja2 \
    beautifulsoup4 \
    lxml \
    pyyaml \
    toml

# Create non-root user
RUN groupadd -r sandbox && useradd -r -g sandbox -d /app -s /bin/bash sandbox

# Set up workspace
RUN mkdir -p /app/workspace && chown -R sandbox:sandbox /app

WORKDIR /app/workspace

USER sandbox

# Git config for auto-commits
RUN git config --global user.email "kutai@local" && \
    git config --global user.name "Kutai" && \
    git config --global init.defaultBranch main

CMD ["sleep", "infinity"]
