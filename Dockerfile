FROM python:3.13-slim

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Copy source code
COPY . .

# Install dependencies
RUN uv sync --frozen --no-dev

# Expose port (Railway sets PORT env var automatically)
EXPOSE 7860

# Run the Gradio app
CMD ["uv", "run", "python", "app.py"]
