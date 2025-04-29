# Use NVIDIA CUDA base image with development tools
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=all

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy your script and data
COPY pymc-bnn.py .
COPY X_train.csv .
COPY X_test.csv .
COPY y_train.csv .
COPY y_test.csv .

# Set environment variables for GPU acceleration
ENV THEANO_FLAGS="device=cuda,floatX=float32"
ENV PYMC_NUMBA=1

# Set the command to run your script
CMD ["python3", "pymc-bnn.py"] 