# ===========================================================================
# Stage 1: Build environment (compile wheels including pyopenjtalk)
# ===========================================================================
FROM docker.io/openvino/ubuntu24_runtime:latest AS builder

USER root

# Install build tools and header files needed for native extensions (e.g. pyopenjtalk)
RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake \
    build-essential \
    python3-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

COPY requirements.txt .
# Compile and package wheels for all requirements
RUN pip wheel --no-cache-dir --wheel-dir=/build/wheels -r requirements.txt

# ===========================================================================
# Stage 2: Clean runtime environment
# ===========================================================================
FROM docker.io/openvino/ubuntu24_runtime:latest

USER root

# Install runtime dependencies (pip and espeak-ng phonemizer for Kokoro multilingual)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    espeak-ng \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install pre-built wheels from builder stage
COPY --from=builder /build/wheels /wheels
COPY requirements.txt .
RUN pip install --no-cache-dir --break-system-packages --no-index --find-links=/wheels -r requirements.txt \
    && rm -rf /wheels

# Copy application source code, translations, and entrypoint
COPY i18n.py wyoming-tts.py wyoming-tts.sh ./
COPY lang ./lang
RUN chmod +x wyoming-tts.sh && ln -s /app/wyoming-tts.sh /usr/local/bin/wyoming-tts

EXPOSE 10021

ENTRYPOINT ["/app/wyoming-tts.sh"]
