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
# Filter out PyTorch and Kokoro (which pulls torch and transformers) to produce a lightweight build (~2GB smaller).
# Multilingual G2P phonemization falls back smoothly to misaki + espeak-ng without needing PyTorch.
RUN grep -v -E '^\s*(torch|kokoro)(\s*([<>=!~].*)?)$' requirements.txt > /build/requirements-no-torch.txt
# Compile and package wheels for lightweight requirements
RUN pip wheel --no-cache-dir --wheel-dir=/build/wheels -r /build/requirements-no-torch.txt

# ===========================================================================
# Stage 2: Clean runtime environment (Lightweight, No PyTorch)
# ===========================================================================
FROM docker.io/openvino/ubuntu24_runtime:latest

USER root

# Install runtime dependencies (espeak-ng phonemizer for Kokoro multilingual)
RUN apt-get update && apt-get install -y --no-install-recommends \
    espeak-ng \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install pre-built wheels from builder stage
COPY --from=builder /build/wheels /wheels
COPY --from=builder /build/requirements-no-torch.txt ./requirements.txt
RUN pip install --no-cache-dir --break-system-packages --no-index --find-links=/wheels -r requirements.txt \
    && rm -rf /wheels

# Copy application source code, translations, and entrypoint
COPY i18n.py wyoming-tts.py wyoming-tts.sh ./
COPY lang ./lang
RUN chmod +x wyoming-tts.sh && ln -s /app/wyoming-tts.sh /usr/local/bin/wyoming-tts

EXPOSE 10021

ENTRYPOINT ["/app/wyoming-tts.sh"]
