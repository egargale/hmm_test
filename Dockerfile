# HMM Futures Analysis Docker Image
# Multi-stage build for optimized production image

# Build stage
FROM python:3.11-slim as builder

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv for faster package management
RUN pip install uv

# Set working directory
WORKDIR /app

# Copy project files
COPY pyproject.toml uv.lock ./
COPY src/ ./src/
COPY cli_comprehensive.py cli_simple.py ./
COPY README.md ./

# Install dependencies
RUN uv sync --frozen --no-dev

# Production stage
FROM python:3.11-slim as production

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH"

# Install runtime system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r hmmuser && useradd -r -g hmmuser hmmuser

# Copy uv and virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /app /app

# Set working directory
WORKDIR /app

# Change ownership to non-root user
RUN chown -R hmmuser:hmmuser /app /opt/venv

# Switch to non-root user
USER hmmuser

# Expose port for potential web interfaces
EXPOSE 8000

# Set default command
CMD ["python", "cli_comprehensive.py", "--help"]

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python cli_comprehensive.py version || exit 1

# Labels
LABEL maintainer="HMM Futures Analysis Team" \
      version="1.0.0" \
      description="HMM Futures Analysis - Comprehensive market regime detection and backtesting" \
      org.opencontainers.image.title="HMM Futures Analysis" \
      org.opencontainers.image.description="Comprehensive HMM-based futures market analysis system" \
      org.opencontainers.image.version="1.0.0" \
      org.opencontainers.image.vendor="HMM Futures Analysis Team"