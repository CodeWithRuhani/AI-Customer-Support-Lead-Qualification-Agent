# ---------------------------------------------------------------
# AI Customer Support & Lead Qualification Agent
# Reproducible container build - fixes Python version, dependency
# resolution, and path issues once and for all.
# ---------------------------------------------------------------
FROM python:3.12-slim

# Prevents .pyc files and forces stdout/stderr to be unbuffered
# (so `docker logs` shows uvicorn output immediately)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies first (separate layer) so Docker can cache this
# step and skip it on rebuilds unless requirements.txt actually changes
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Now copy the rest of the application code
COPY . .

# The SQLite file is created at runtime from schema.sql; persisting it
# across container restarts is handled via a volume (see docker-compose.yml)
EXPOSE 8000

# Basic container-level healthcheck hitting the root endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/')" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
