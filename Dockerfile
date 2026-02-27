FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY models/ models/
COPY webapp/ webapp/
COPY ["2026 Github ARA Pond IDs Key.csv", "."]

# Tell the app where to find models and data
ENV APP_BASE_DIR=/app
# Cloud Run sets PORT env var (default 8080)
ENV PORT=8080

# Run with gunicorn for production
CMD exec gunicorn --bind :$PORT --workers 2 --timeout 120 --chdir webapp app:app
