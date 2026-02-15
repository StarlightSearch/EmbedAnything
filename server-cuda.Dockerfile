FROM nvidia/cuda:12.2.0-devel-ubuntu22.04 AS base-builder

ENV SCCACHE=0.10.0
ENV RUSTC_WRAPPER=/usr/local/bin/sccache
ENV PATH="/root/.cargo/bin:${PATH}"
# aligned with `cargo-chef` version in `lukemathwalker/cargo-chef:latest-rust-1.85-bookworm`
ENV CARGO_CHEF=0.1.71
ENV CUDA_COMPUTE_CAP=75

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    curl \
    libssl-dev \
    pkg-config \
    python3 \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Download and configure sccache
RUN curl -fsSL https://github.com/mozilla/sccache/releases/download/v$SCCACHE/sccache-v$SCCACHE-x86_64-unknown-linux-musl.tar.gz | tar -xzv --strip-components=1 -C /usr/local/bin sccache-v$SCCACHE-x86_64-unknown-linux-musl/sccache && \
    chmod +x /usr/local/bin/sccache

RUN curl https://sh.rustup.rs -sSf | bash -s -- -y
RUN cargo install cargo-chef --version $CARGO_CHEF --locked

FROM base-builder AS planner

WORKDIR /app

COPY processors processors
COPY rust rust
COPY python python
COPY server server
COPY Cargo.toml ./
COPY Cargo.lock ./

RUN cargo chef prepare --recipe-path recipe.json

FROM base-builder AS builder

ARG GIT_SHA
ARG DOCKER_LABEL

# sccache specific variables
ARG SCCACHE_GHA_ENABLED

# Limit parallelism
ARG RAYON_NUM_THREADS=4
ARG CARGO_BUILD_JOBS
ARG CARGO_BUILD_INCREMENTAL

WORKDIR /app

COPY --from=planner /app/recipe.json recipe.json

RUN --mount=type=secret,id=actions_results_url,env=ACTIONS_RESULTS_URL \
    --mount=type=secret,id=actions_runtime_token,env=ACTIONS_RUNTIME_TOKEN \
    cargo chef cook --release --recipe-path recipe.json --package server --features cuda && sccache -s;

COPY processors processors
COPY rust rust
COPY server server
COPY Cargo.toml ./
COPY Cargo.lock ./

RUN --mount=type=secret,id=actions_results_url,env=ACTIONS_RESULTS_URL \
    --mount=type=secret,id=actions_runtime_token,env=ACTIONS_RUNTIME_TOKEN \
    cargo build --release --bin server --features cuda && sccache -s;

RUN strip /app/target/release/server

FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04 AS runtime

WORKDIR /app

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    ca-certificates \
    libssl-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/target/release/server /usr/local/bin/server

EXPOSE 8080

CMD ["server"]
