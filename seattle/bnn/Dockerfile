# Base image with CUDA 12 and Python
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# Set environment variables to prevent Python from writing .pyc files
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-venv \
    python3-pip \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set python3.10 as default
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# Upgrade pip
RUN pip install --upgrade pip

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your project code into the container
WORKDIR /app
COPY . /app

# Default command (optional override with docker run)
CMD ["python", "bnn/bnn_seattle.py"]