FROM python:3.14-slim

WORKDIR /app


# Copy dependency files first (better caching)
COPY requirements.txt pyproject.toml ./

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy rest of the application
COPY . .


EXPOSE 7860

# Run FastAPI app
CMD ["python", "main.py"]