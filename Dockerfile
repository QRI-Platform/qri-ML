FROM python:3.14-slim

# Install uv binary from official image (no curl or sudo required)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Copy dependency files first for layer caching
COPY pyproject.toml uv.lock ./

# Install project dependencies
RUN uv sync --frozen

# Copy rest of the application
COPY . .

EXPOSE 7860

# Run application with uv
CMD ["uv", "run", "main.py"]