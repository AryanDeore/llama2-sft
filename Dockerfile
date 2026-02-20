FROM python:3.11-slim

WORKDIR /app

# Install CPU-only PyTorch first (cached layer, biggest download)
RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
RUN pip install --no-cache-dir \
    gradio>=5.0.0 \
    huggingface_hub>=0.30.0 \
    sentencepiece>=0.2.0

# Copy only the files needed for serving
COPY app.py generate.py checkpoint.py ./
COPY models/ models/
COPY utils/ utils/
COPY tokenizer.model ./

EXPOSE 7860

CMD ["python", "app.py"]
