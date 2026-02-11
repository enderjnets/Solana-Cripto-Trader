FROM nvcr.io/nvidia/personaplex:latest

# Install dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create persona directory
RUN mkdir -p /app/persona

# Copy persona files
COPY ./persona/voice.wav /app/persona/voice.wav
COPY ./persona/personality.txt /app/persona/personality.txt

# Expose WebSocket port
EXPOSE 8998

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8998/health || exit 1

CMD ["python3", "-m", "personaplex", "--port", "8998"]
