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

RUN apt-get update
RUN apt-get install libssl-dev pkg-config python3-full python3-pip -y 
RUN pip3 install maturin[patchelf] --break-system-packages
RUN wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | gpg --dearmor | tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null \
    echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | tee /etc/apt/sources.list.d/oneAPI.list \
    apt-get update \
    apt-get install -y intel-oneapi-mkl-devel \
    export LD_LIBRARY_PATH="/opt/intel/oneapi/compiler/2024.2/lib:$LD_LIBRARY_PATH"     

COPY . .
RUN maturin build --release --features mkl,extension-module

FROM python:3.11-slim

WORKDIR /app

COPY --from=builder /app/target/wheels .

COPY . .

RUN pip install *.whl

RUN pip install numpy pillow pytest

CMD ["pytest"]
