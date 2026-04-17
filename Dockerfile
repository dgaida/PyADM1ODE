# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies including Mono for .NET support
RUN apt-get update && apt-get install -y --no-install-recommends \
    mono-complete \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt /app/
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the project files
COPY . /app/

# Install the package in editable mode
RUN pip install -e .

# Expose port if needed (e.g., for Jupyter)
EXPOSE 8888

# Create a directory for outputs
RUN mkdir -p output

# Default command: run the basic digester example
CMD ["python", "examples/01_basic_digester.py"]
