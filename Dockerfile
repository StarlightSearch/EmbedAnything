FROM lukemathwalker/cargo-chef:latest-rust-1 AS chef
WORKDIR /app

FROM chef AS planner
COPY . .
RUN cargo chef prepare --recipe-path recipe.json

FROM chef AS builder 
COPY --from=planner /app/recipe.json recipe.json
# Build dependencies - this is the caching Docker layer!
RUN cargo chef cook --release --recipe-path recipe.json
# Build application

# Download Intel GPG key and add repository
RUN wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | gpg --dearmor | tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null \
    && echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | tee /etc/apt/sources.list.d/oneAPI.list

# Install Intel MKL and extract libiomp5.so
RUN apt-get update \
    && apt-get install -y intel-oneapi-mkl-devel \
    && cp /opt/intel/oneapi/compiler/2024.2/lib/libiomp5.so /app/libiomp5.so

RUN apt-get install libssl-dev pkg-config python3-full python3-pip -y 
RUN pip3 install maturin[patchelf] --break-system-packages

COPY . .
RUN maturin build --release --features mkl,extension-module

FROM python:3.11-slim

WORKDIR /app

# Copy the extracted libiomp5.so from the builder stage
COPY --from=builder /app/libiomp5.so /usr/lib/

# Set the library path
ENV LD_LIBRARY_PATH="/usr/lib:$LD_LIBRARY_PATH"

COPY . .

RUN pip install target/wheels/*.whl

RUN pip install numpy pillow pytest

CMD ["pytest"]
