FROM python:3.12-slim

# Install system deps (git for the workspace, plus common build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    build-essential \
    ca-certificates \
    gnupg \
    && rm -rf /var/lib/apt/lists/*

# ── Node.js 20 LTS ──
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/* \
    && npm install -g eslint prettier typescript jest

# ── Go 1.22 ──
RUN curl -fsSL https://go.dev/dl/go1.22.8.linux-amd64.tar.gz | tar -C /usr/local -xzf -
ENV PATH="/usr/local/go/bin:/root/go/bin:${PATH}"

# ── Rust (minimal, installed system-wide) ──
ENV RUSTUP_HOME="/usr/local/rustup" CARGO_HOME="/usr/local/cargo"
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --profile minimal --default-toolchain stable \
    && chmod -R a+rX /usr/local/rustup /usr/local/cargo
ENV PATH="/usr/local/cargo/bin:${PATH}"
RUN rustup component add clippy rustfmt

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
    pytest-cov \
    mypy \
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
