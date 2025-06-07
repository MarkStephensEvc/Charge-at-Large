# Base Stage: Install common system dependencies
# Use an official Python runtime as a parent image
FROM python:3.12-slim AS base

# Set the working directory in the container
WORKDIR /app

# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install system dependencies and Python packages
RUN apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends \
    libgdal-dev \
    gdal-bin \
    libspatialindex-dev \
    libproj-dev \
    libglib2.0-0 \
    libgl1-mesa-glx \
    libxrender1 \
    libxext6 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Builder Stage: Build wheels
FROM base AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
build-essential \
gcc \
&& apt-get clean && rm -rf /var/lib/apt/lists/*
# Copy only requirements first for efficient layer caching

RUN pip install --upgrade pip wheel

COPY requirements.txt /app/

# Pre-build wheels for all packages in requirements.txt
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /wheels -r /app/requirements.txt

# Runtime Stage: Final image
FROM base

# Add a non-root user for better security
RUN useradd -ms /bin/bash appuser

COPY --from=builder /wheels /wheels
 #Copy requirements.txt explicitly to the runtime stage
COPY requirements.txt /app/ 
RUN pip install --no-cache-dir --find-links=/wheels -r /app/requirements.txt

COPY . /app

# Set permissions and switch to non-root user
RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 3838
CMD ["python", "app.py", "--host=0.0.0.0", "--port=3838"]