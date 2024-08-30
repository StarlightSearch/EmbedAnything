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
COPY . .
RUN maturin build --release

FROM python:3.11-slim

WORKDIR /app

COPY --from=builder /app/target/wheels .

COPY . .

RUN pip install *.whl

RUN pip install numpy pillow pytest

CMD ["python", "examples/clip.py"]
