FROM nvidia/cuda:12.8.1-runtime-ubuntu24.04

# Python 3.11 from deadsnakes PPA (not in Ubuntu 24.04 default repos)
RUN apt-get update && apt-get install -y --no-install-recommends software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3.11-dev \
    tesseract-ocr \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    fonts-dejavu-core \
    libxcb-xinerama0 \
    libxcb-cursor0 \
    libxkbcommon-x11-0 \
    libxcb-icccm4 \
    libxcb-keysyms1 \
    libxcb-shape0 \
    libxcb-render-util0 \
    libsm6 \
    libice6 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Copy dependency files first for better caching
COPY pyproject.toml uv.lock .python-version ./

# Install Python dependencies
RUN uv sync --frozen --no-install-project

# Copy project source
COPY racing_tools/ racing_tools/

# Install the project itself
RUN uv sync --frozen

ENTRYPOINT ["uv", "run"]
