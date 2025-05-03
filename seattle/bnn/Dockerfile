# Base image with CUDA 12.1 and Python
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# Set environment variables
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

# Create symlink for 'python' command
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# Upgrade pip, setuptools, and wheel
RUN pip install --upgrade pip setuptools wheel

# Optional: Install Jupyter & Jupytext for interactive dev
RUN pip install jupyterlab jupytext

# Set working directory
WORKDIR /app

# Copy requirements first (for caching)
# Install requirements
COPY requirements.txt /app/requirements.txt
WORKDIR /app
RUN pip install --no-cache-dir -r requirements.txt

# Install CUDA-enabled JAX
RUN pip install --upgrade \
  jax==0.4.25 \
  jaxlib==0.4.25+cuda12.cudnn89 \
  numpyro==0.13.2 \
  -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
# Copy rest of the code
COPY . /app

# Set default command to run BNN
CMD ["python", "bnn.py"]