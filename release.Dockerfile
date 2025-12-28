# Use NVIDIA CUDA base image
FROM nvidia/cuda:12.2.2-devel-ubuntu22.04

# Set non-interactive mode and timezone
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Set working directory
WORKDIR /app

# Update package list and install basic dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    software-properties-common \
    tzdata \
    && rm -rf /var/lib/apt/lists/*

# Add deadsnakes PPA and install Python versions
RUN add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update && \
    apt-get install -y \
    python3.10 \
    python3.11 \
    python3.12 \
    python3.13 \
    python3.14 \
    python3.12-venv \
    python3.11-venv \
    python3.13-venv \
    python3.14-venv \
    python3-pip \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Clone the repository and checkout cuda branch
RUN git clone https://github.com/StarlightSearch/EmbedAnything.git && \
    cd EmbedAnything && \
    git checkout cuda

# Set working directory to the cloned repo
WORKDIR /app/EmbedAnything

# Create Python virtual environment and install dependencies
RUN python3.11 -m venv /opt/embed_env && \
    /opt/embed_env/bin/pip install --upgrade pip && \
    /opt/embed_env/bin/pip install maturin[patchelf] auditwheel

# Activate virtual environment by default
ENV PATH="/opt/embed_env/bin:$PATH"
