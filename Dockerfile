FROM python:3.11-slim

WORKDIR /app

# Install system dependencies including build tools
RUN apt-get update && apt-get install -y \
    gdal-bin \
    libgdal-dev \
    git \
    build-essential \
    g++ \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements files
COPY requirements_api.txt ./

# Install Python dependencies (API only - much faster!)
RUN pip install --no-cache-dir -r requirements_api.txt

# Copy application code
COPY src/ ./src/
COPY outputs/ ./outputs/
COPY data/ ./data/
COPY mlflow_config.py ./

# Create necessary directories
RUN mkdir -p logs mlruns

# Expose ports
EXPOSE 8000 8501 3000 5000

# Default command (can be overridden)
CMD ["python", "src/api/prediction_api.py"]
