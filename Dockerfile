FROM python:3.11-slim

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

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
