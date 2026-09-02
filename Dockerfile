FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libxcb1 \
    && rm -rf /var/lib/apt/lists/*
    
# Install uv binary from official image (no curl or sudo required)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Copy dependency files first for layer caching
COPY pyproject.toml uv.lock ./

# Install project dependencies
RUN uv sync --frozen --no-cache

# Copy rest of the application
COPY . .

EXPOSE 7860

# Run application with uv
CMD ["uv", "run", "start"]
