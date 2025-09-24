# Multi-stage Docker build for ML Drilling Operations

# Stage 1: Base Python environment
FROM python:3.10-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r drilling && useradd -r -g drilling drilling

# Stage 2: Dependencies installation
FROM base as dependencies

# Set working directory
WORKDIR /app

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Stage 3: Application
FROM dependencies as application

# Copy application code
COPY . .

# Install the package
RUN pip install -e .

# Create necessary directories
RUN mkdir -p data/raw data/processed models outputs/logs outputs/reports outputs/figures

# Change ownership to non-root user
RUN chown -R drilling:drilling /app

# Switch to non-root user
USER drilling

# Expose ports
EXPOSE 8000 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["python", "run_pipeline.py", "--mode", "api"]

# Stage 4: API-only lightweight image
FROM dependencies as api-only

# Copy only API-related files
COPY src/api/ ./src/api/
COPY src/models/ ./src/models/
COPY src/data/ ./src/data/
COPY src/utils/ ./src/utils/
COPY setup.py .
COPY run_pipeline.py .

# Install the package
RUN pip install -e .

# Create directories
RUN mkdir -p models outputs/logs

# Change ownership
RUN chown -R drilling:drilling /app

# Switch to non-root user  
USER drilling

# Expose API port only
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# API command
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]

# Stage 5: Dashboard-only image
FROM dependencies as dashboard-only

# Copy only dashboard-related files
COPY src/visualization/ ./src/visualization/
COPY src/models/ ./src/models/
COPY src/data/ ./src/data/
COPY src/utils/ ./src/utils/
COPY setup.py .

# Install the package
RUN pip install -e .

# Create directories
RUN mkdir -p data/raw data/processed models

# Change ownership
RUN chown -R drilling:drilling /app

# Switch to non-root user
USER drilling

# Expose dashboard port only
EXPOSE 8501

# Dashboard command
CMD ["streamlit", "run", "src/visualization/dashboard.py", "--server.port", "8501", "--server.address", "0.0.0.0"]