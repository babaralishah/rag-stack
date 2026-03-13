FROM python:3.11-slim

WORKDIR /app

# System deps (lightweight)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# Install python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Expose API
EXPOSE 8000

# Run FastAPI
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
