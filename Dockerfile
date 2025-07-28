# Use Python 3.9 slim image for AMD64 architecture
FROM --platform=linux/amd64 python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies with error handling
RUN pip install --no-cache-dir -r requirements.txt || \
    (echo "Failed to install requirements, trying with --force-reinstall" && \
     pip install --no-cache-dir --force-reinstall -r requirements.txt)

# Copy the sample dataset for training
COPY sample_dataset/ ./sample_dataset/

# Copy application files
COPY pdf_outline_extractor.py .
COPY precise_outline_extractor.py .
COPY app.py .

# Make app.py executable (ignore errors on Windows)
RUN chmod +x app.py || true

# Create input and output directories
RUN mkdir -p /app/input /app/output

# Set environment variables for better error handling
ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=utf-8

# Set the entrypoint
ENTRYPOINT ["python", "app.py"]
