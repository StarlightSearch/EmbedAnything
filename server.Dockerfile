FROM lukemathwalker/cargo-chef:latest AS chef
WORKDIR /app    
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    python3 \
    python3-dev


FROM chef AS planner
COPY . .
RUN cargo chef prepare --recipe-path recipe.json

FROM chef AS builder
COPY --from=planner /app/recipe.json recipe.json
# Build dependencies - this is the caching Docker layer!
RUN cargo chef cook --release --recipe-path recipe.json --package server
# Build application
COPY . .
RUN cargo build --release --package server
RUN strip target/release/server

# We do not need the Rust toolchain to run the binary!
FROM debian:trixie-slim AS runtime
WORKDIR /app

# # Combine RUN commands and cleanup in the same layer


COPY --from=builder /app/target/release/server /usr/local/bin

EXPOSE 8080

CMD ["server"]