# Use a pre-built Docker image with cargo-chef and the Rust toolchain
# the cargo-shef should use a Bookworm version. Alpine does not compile
FROM lukemathwalker/cargo-chef:latest-rust-1.90-bookworm AS chef
WORKDIR /app

# Prepare the build environment using cargo-chef
FROM chef AS planner
COPY . .
RUN cargo chef prepare --recipe-path recipe.json

# Cook the build dependencies using cargo-chef
FROM chef AS builder 
WORKDIR /app
COPY --from=planner /app/recipe.json recipe.json
# Build dependencies - this is the caching Docker layer!
RUN cargo chef cook --release --recipe-path recipe.json

# Build the application
COPY . .
RUN cargo build --release --bin midna
RUN ls -l /app/target/release

# Base image for the final application
FROM ubuntu:noble

# Update package lists and install wget
RUN apt-get update \
  && apt-get install -y wget \
  && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /usr/local/bin
COPY .env .
COPY assets assets

# Download and install ONNX Runtime binary release
RUN wget https://github.com/microsoft/onnxruntime/releases/download/v1.8.1/onnxruntime-linux-x64-1.8.1.tgz \
  && tar -xzf onnxruntime-linux-x64-1.8.1.tgz \
  && mv onnxruntime-linux-x64-1.8.1 /opt/onnxruntime \
  && ldconfig /opt/onnxruntime/lib

# Copy the compiled Rust binary to the final image
COPY --from=builder /app/target/release/midna /usr/local/bin

# Command to run the application
ENTRYPOINT ["/usr/local/bin/midna"]
