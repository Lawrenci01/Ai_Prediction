# ============================================================
# AI Prediction API — Dockerfile
# FastAPI + TensorFlow + MySQL (aiomysql)
# ============================================================

FROM python:3.11-slim

# System dependencies for TensorFlow + aiomysql
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libssl-dev \
    pkg-config \
    default-libmysqlclient-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy and install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Render uses the PORT env variable — default to 8000
ENV PORT=8000

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PORT}/ || exit 1

# Start the FastAPI app with uvicorn
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --workers 1