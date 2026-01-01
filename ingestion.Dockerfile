FROM ghcr.io/astral-sh/uv:0.4.30-python3.13-slim

WORKDIR /app

# Copy the ingestion script and its dependencies
COPY ml/ingestion/ /app/

# Sync dependencies
RUN uv sync

# Run the script
CMD ["uv", "run", "python", "script.py"]
