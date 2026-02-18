FROM python:3.13-slim

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Copy source code
COPY . .

# Install dependencies, then swap in CPU-only PyTorch (saves ~2.5 GB)
RUN uv sync --frozen --no-dev && \
    uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cpu

# Expose port (Railway sets PORT env var automatically)
EXPOSE 7860

# Run the Gradio app
CMD ["uv", "run", "python", "app.py"]
