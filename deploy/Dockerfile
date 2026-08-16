# EVIDENCE-Net Production Reference Container (Phase 15)

# Stage 1: Build Frontend Assets
FROM node:20-alpine AS frontend-builder
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci --ignore-scripts
COPY frontend/ ./
RUN npm run build

# Stage 2: Python Service Runtime
FROM python:3.11-slim AS runner
WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy python packaging files & install evidence-net
COPY pyproject.toml README.md ./
COPY src/ src/
RUN pip install --no-cache-dir -e .

# Copy built frontend assets to frontend/dist for FastAPI static mounting
COPY --from=frontend-builder /app/frontend/dist /app/frontend/dist

# Copy deployment configurations
COPY configs/ configs/

# Environment variables
ENV PORT=8000
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Launch FastAPI service
CMD ["python", "-m", "evidence_net.api.app"]
