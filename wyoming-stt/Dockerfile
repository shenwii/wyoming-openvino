FROM docker.io/openvino/ubuntu24_runtime:latest

USER root

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --break-system-packages -r requirements.txt

# Copy application source code, translations, and entrypoint
COPY i18n.py wyoming-stt.py wyoming-stt.sh ./
COPY lang ./lang
RUN chmod +x wyoming-stt.sh && ln -s /app/wyoming-stt.sh /usr/local/bin/wyoming-stt

EXPOSE 10020

ENTRYPOINT ["/app/wyoming-stt.sh"]
