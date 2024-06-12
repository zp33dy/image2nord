# Stage 1: Build the application
FROM ubuntu:noble
FROM rust:latest

# Install dependencies
RUN apt-get update && apt-get install -y \
    libopencv-dev \
    pkg-config \
    cmake \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /usr/src/app

# Install ONNX Runtime using apt
RUN apt-get install -y onnxruntime 

# Install Rust dependencies
RUN rustup update && \
    rustup component add rustfmt && \
    rustup component add clippy



# Copy the Cargo.toml and Cargo.lock files
COPY Cargo.toml ./

# Copy .env file
COPY .env ./.env
# Copy the source code
COPY src ./src

# Build the Rust application
RUN cargo install --path .

CMD ["image2nord"]
