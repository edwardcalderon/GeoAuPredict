FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gdal-bin \
    libgdal-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy environment file
COPY environment.yml .

# Install conda and dependencies
RUN pip install --no-cache-dir conda && \
    conda env create -f environment.yml

# Copy application code
COPY . .

# Activate conda environment
SHELL ["conda", "run", "-n", "geoau", "/bin/bash", "-c"]

# Expose port for web application
EXPOSE 3000

# Default command
CMD ["conda", "run", "-n", "geoau", "python", "-m", "src.main"]
