# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# To request access to the AlphaFold 3 model parameters, follow the process set
# out at https://github.com/google-deepmind/alphafold3. You may only use these
# if received directly from Google. Use is subject to terms of use available at
# https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md
#
# Modifications Copyright 2026 Romero Lab, Duke University

# syntax=docker/dockerfile:1
FROM nvidia/cuda:13.0.0-base-ubuntu24.04

# This Dockerfile uses MMseqs2-GPU for protein MSA searches instead of HMMER.

# Ye addition - remove apt source built-in
RUN rm /etc/apt/sources.list.d/ubuntu.sources

# Ye addition - changing Ubuntu apt source mirror
RUN <<'EOF'
cat > /etc/apt/sources.list << 'SOURCES'
# 默认注释了源码镜像以提高 apt update 速度，如有需要可自行取消注释
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ noble main restricted universe multiverse
# deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ noble main restricted universe multiverse
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ noble-updates main restricted universe multiverse
# deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ noble-updates main restricted universe multiverse
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ noble-backports main restricted universe multiverse
# deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ noble-backports main restricted universe multiverse

# 以下安全更新软件源为官方源配置
# deb http://security.ubuntu.com/ubuntu/ noble-security main restricted universe multiverse
# deb-src http://security.ubuntu.com/ubuntu/ noble-security main restricted universe multiverse

# 预发布软件源，不建议启用
# deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ noble-proposed main restricted universe multiverse
# # deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ noble-proposed main restricted universe multiverse
SOURCES
EOF

# Install system packages
RUN DEBIAN_FRONTEND=noninteractive \
    apt-get update --quiet \
    && apt-get install --yes --quiet python3.12 python3.12-dev \
    && apt-get install --yes --quiet git wget gcc g++ make zlib1g-dev zstd pigz \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:0.9.24 /uv /uvx /bin/

# Ye addition - set uv mirror (override with --build-arg UV_INDEX=... for CI)
ARG UV_INDEX="http://mirrors.aliyun.com/pypi/simple/"
ENV UV_DEFAULT_INDEX=$UV_INDEX

# Disable uv bytecode compilation as this hits ulimit on some systems
ENV UV_COMPILE_BYTECODE=0

# Read local venv
ENV UV_PROJECT_ENVIRONMENT=/alphafold3_venv
RUN uv venv $UV_PROJECT_ENVIRONMENT
ENV PATH="/hmmer/bin:/alphafold3_venv/bin:$PATH"

# Install HMMER from source (matching upstream AF3).
# nhmmer (RNA MSA search fallback) is included in the HMMER suite.
RUN mkdir -p /hmmer_build /hmmer && \
    wget -q http://eddylab.org/software/hmmer/hmmer-3.4.tar.gz -P /hmmer_build && \
    (cd /hmmer_build && echo "ca70d94fd0cf271bd7063423aabb116d42de533117343a9b27a65c17ff06fbf3 hmmer-3.4.tar.gz" | sha256sum --check) && \
    (cd /hmmer_build && tar zxf hmmer-3.4.tar.gz && rm hmmer-3.4.tar.gz) && \
    (cd /hmmer_build/hmmer-3.4 && ./configure --prefix /hmmer) && \
    (cd /hmmer_build/hmmer-3.4 && make -j$(nproc)) && \
    (cd /hmmer_build/hmmer-3.4 && make install) && \
    (cd /hmmer_build/hmmer-3.4/easel && make install) && \
    rm -rf /hmmer_build

# Install MMSeqs2-GPU and Foldseek
RUN wget -q -O mmseqs.tar.gz https://mmseqs.com/latest/mmseqs-linux-gpu.tar.gz & \
    wget -q -O foldseek.tar.gz https://mmseqs.com/foldseek/foldseek-linux-gpu.tar.gz & \
    wait && \
    pigz -dc mmseqs.tar.gz | tar xf - && \
    pigz -dc foldseek.tar.gz | tar xf - && \
    cp mmseqs/bin/mmseqs foldseek/bin/foldseek /usr/local/bin/ && \
    rm -rf mmseqs foldseek *.tar.gz

# Set working directory
WORKDIR /app/alphafold

# Copy dependency files first for better caching
# This layer is cached until pyproject.toml or uv.lock changes
COPY pyproject.toml uv.lock ./

# Install dependencies (without the project itself)
# Using cache mount for faster rebuilds when dependencies change
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --all-groups

# Now copy the full source code
COPY . .

# Install the project itself
# Using --no-editable for production (static package)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --all-groups --no-editable

# Build chemical components database (this binary was installed by uv sync)
RUN DEBIAN_FRONTEND=noninteractive apt-get update --quiet && apt-get install --yes --quiet util-linux && rm -rf /var/lib/apt/lists/*
RUN date && hwclock -s || true
RUN --mount=type=cache,target=/root/.cache/uv \
    uv run --no-sync build_data

# To work around a known XLA issue causing the compilation time to greatly
# increase, the following environment variable setting XLA flags must be enabled
# when running AlphaFold 3. Note that if using CUDA capability 7 GPUs, it is
# necessary to set the following XLA_FLAGS value instead:
# ENV XLA_FLAGS="--xla_disable_hlo_passes=custom-kernel-fusion-rewriter"
# (no need to disable gemm in that case as it is not supported for such GPU).
ENV TF_CPP_MIN_LOG_LEVEL=3
ENV XLA_FLAGS="--xla_gpu_enable_triton_gemm=false"

# Memory settings used for folding up to 5,120 tokens on A100 80 GB.
# When using run_data_pipeline.py separately (recommended for MMseqs2-GPU),
# this can be set to 0.95 for maximum inference performance.
ENV XLA_CLIENT_MEM_FRACTION=0.95

CMD ["uv", "run", "python3", "run_alphafold.py"]
