# Stage 1: Chef base with CUDA development tools
FROM nvidia/cuda:12.2.2-devel-ubuntu22.04 AS chef
WORKDIR /app

# Set non-interactive mode
ENV DEBIAN_FRONTEND=noninteractive
ENV TORCH_CUDA_ARCH_LIST=Turing
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES video,compute,utility
# Set CUDA compute capability for candle-kernels (Turing = 7.5, encoded as 75)
# This bypasses the need for nvidia-smi during build
# Format: major * 10 + minor (e.g., 7.5 -> 75, 8.0 -> 80, 8.6 -> 86)
ENV CUDA_COMPUTE_CAP=75

# Install build dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    python3 \
    python3-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Install cargo-chef (we use CUDA base image instead of lukemathwalker/cargo-chef 
# because we need CUDA development tools)
RUN cargo install cargo-chef --locked

# Create a mock nvidia-smi script as fallback (in case CUDA_COMPUTE_CAP doesn't work)
# This script will be used if nvidia-smi is not available
RUN echo '#!/bin/bash\n\
# Mock nvidia-smi for Docker build\n\
# Returns compute capability 75 (Turing 7.5) for build-time detection\n\
if echo "$*" | grep -q "compute_cap"; then\n\
    echo "compute_cap"\n\
    echo "75"\n\
elif echo "$*" | grep -q "query"; then\n\
    # Handle --query-gpu format\n\
    if echo "$*" | grep -q "csv"; then\n\
        echo "compute_cap"\n\
        echo "75"\n\
    else\n\
        echo "CUDA Version: 12.2"\n\
        echo "Driver Version: 535.00"\n\
        echo "Compute Capability: 7.5"\n\
    fi\n\
else\n\
    # Default output\n\
    echo "NVIDIA-SMI 535.00"\n\
    echo "Driver Version: 535.00"\n\
    echo "CUDA Version: 12.2"\n\
fi\n\
exit 0' > /usr/local/bin/nvidia-smi && chmod +x /usr/local/bin/nvidia-smi

# Stage 2: Planner - prepare recipe
FROM chef AS planner
COPY . .
RUN cargo chef prepare --recipe-path recipe.json

# Stage 3: Builder - cook dependencies and build
FROM chef AS builder
COPY --from=planner /app/recipe.json recipe.json
# Build dependencies - this is the caching Docker layer!
RUN cargo chef cook --release --recipe-path recipe.json --package server --features cuda
# Build application
COPY . .
RUN cargo build --release -p server --features cuda
RUN strip target/release/server

# Stage 4: Runtime - minimal CUDA runtime image
FROM nvidia/cuda:12.2.2-runtime-ubuntu22.04 AS runtime
WORKDIR /app

# Install minimal runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy the stripped binary from builder
COPY --from=builder /app/target/release/server /usr/local/bin/server

EXPOSE 8080

CMD ["server"]

