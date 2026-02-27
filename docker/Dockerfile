# FlashInfer-Bench Docker Image
# GPU kernel benchmarking and development environment
#
# Build:
#   docker build -t flashinfer-bench:cu131 .
#   docker build --build-arg CUDA_VERSION=12.9.1 -t flashinfer-bench:cu129 .
#   docker build --build-arg CUDA_VERSION=13.0.1 -t flashinfer-bench:cu130 .

ARG CUDA_VERSION=13.1.1
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu24.04

ARG CUDA_VERSION
ARG FLASHINFER_VERSION=0.6.1

ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda \
    PATH="${PATH}:/usr/local/cuda/bin" \
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/cuda/lib64"

# ============================================================
# System dependencies + Python 3.12
# ============================================================
RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && apt-get install -y --no-install-recommends \
    # Python
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.12-full \
    python3.12-dev \
    wget \
    # Core utilities
    ca-certificates \
    curl \
    git \
    git-lfs \
    unzip \
    # Build tools
    build-essential \
    cmake \
    ninja-build \
    ccache \
    # Development
    vim \
    tmux \
    zsh \
    htop \
    tree \
    less \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 \
    && update-alternatives --set python3 /usr/bin/python3.12 \
    && ln -sf /usr/bin/python3.12 /usr/bin/python \
    && wget -q https://bootstrap.pypa.io/get-pip.py \
    && python3 get-pip.py --break-system-packages \
    && rm get-pip.py \
    && python3 -m pip config set global.break-system-packages true \
    && rm -rf /var/lib/apt/lists/*

# ============================================================
# CUDA Toolkit (includes NCU, Nsys, Sanitizer, cuda-gdb, etc.)
# ============================================================
RUN --mount=type=cache,target=/var/cache/apt \
    CUDA_VER="${CUDA_VERSION}" \
    && CUDA_MAJOR="${CUDA_VER%%.*}" \
    && CUDA_MINOR="$(echo "${CUDA_VER}" | cut -d. -f2)" \
    && apt-get update \
    && apt-get install -y \
    cuda-toolkit-${CUDA_MAJOR}-${CUDA_MINOR} \
    && rm -rf /var/lib/apt/lists/*

# ============================================================
# Python packages
# ============================================================
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --upgrade pip setuptools wheel \
    # Fast package manager
    && python3 -m pip install uv \
    # IPython for interactive development
    && python3 -m pip install ipython

# Install PyTorch and FlashInfer
RUN --mount=type=cache,target=/root/.cache/pip \
    case "$CUDA_VERSION" in \
        12.6.*) CUINDEX=126 ;; \
        12.8.*) CUINDEX=128 ;; \
        12.9.*) CUINDEX=129 ;; \
        13.0.*) CUINDEX=130 ;; \
        13.1.*) CUINDEX=131 ;; \
        *) echo "Unsupported CUDA version: $CUDA_VERSION" && exit 1 ;; \
    esac \
    && python3 -m pip install torch --index-url https://download.pytorch.org/whl/cu${CUINDEX} \
    && python3 -m pip install flashinfer-python==${FLASHINFER_VERSION} flashinfer-cubin \
    && python3 -m pip install flashinfer-jit-cache --index-url https://flashinfer.ai/whl/cu${CUINDEX}

# Install flashinfer-bench (editable install will be done via volume mount)
# For now, install dependencies
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install \
    pydantic>=2.0.0 \
    safetensors>=0.5.0 \
    apache-tvm-ffi>=0.1.2 \
    docstring-parser>=0.16 \
    # Dev dependencies
    pytest>=7.0.0 \
    pytest-cov>=4.0.0 \
    black>=22.0.0 \
    isort>=5.0.0 \
    ruff>=0.1.0 \
    pre-commit>=3.0.0 \
    setuptools

# ============================================================
# Shell configuration
# ============================================================
RUN sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" "" --unattended \
    && git clone --depth 1 https://github.com/zsh-users/zsh-autosuggestions ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-autosuggestions \
    && git clone --depth 1 https://github.com/zsh-users/zsh-syntax-highlighting.git ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-syntax-highlighting \
    && sed -i 's/plugins=(git)/plugins=(git zsh-autosuggestions zsh-syntax-highlighting)/' ~/.zshrc

# Set working directory
WORKDIR /workspace

# Default command
CMD ["/bin/zsh"]
