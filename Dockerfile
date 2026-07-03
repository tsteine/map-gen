ARG RUST_TOOLCHAIN=1.96.1
FROM rust:${RUST_TOOLCHAIN}-slim-bookworm AS rust_toolchain

FROM nvidia/cuda:13.0.0-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    VIRTUAL_ENV=/opt/map-gen \
    PATH=/opt/map-gen/bin:/usr/local/cargo/bin:$PATH \
    PYTHONPATH=/workspace/python \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        build-essential \
        git \
        pkg-config \
        python3.12 \
        python3.12-dev \
        python3.12-venv \
    && rm -rf /var/lib/apt/lists/*

RUN python3.12 -m venv ${VIRTUAL_ENV} \
    && python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir \
        "maturin>=1.7,<2.0" \
        flask \
        ipython \
        numpy \
        pydantic \
        safetensors \
    && python -m pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu130

COPY --from=rust_toolchain /usr/local/cargo /usr/local/cargo
COPY --from=rust_toolchain /usr/local/rustup /usr/local/rustup

ENV CARGO_HOME=/usr/local/cargo \
    RUSTUP_HOME=/usr/local/rustup

WORKDIR /workspace
COPY . .

RUN maturin develop --release

ARG MODEL_FILENAME=2026-07-01T22:39:14.683967-zebes-testing-round_1400.safetensors

RUN mkdir -p models \
    && curl -fL \
        -o "models/${MODEL_FILENAME}" \
        "https://f004.backblazeb2.com/file/map-rando-artifacts/map-gen-models/${MODEL_FILENAME}"

CMD ["python", "python/serve.py", "--help"]
