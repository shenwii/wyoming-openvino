FROM docker.io/openvino/ubuntu24_runtime:latest

USER root

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --break-system-packages -r requirements.txt

# Copy application source code, default config, translations, and entrypoint
COPY i18n.py wyoming-wakeword.py wakeword_config*.yaml wyoming-wakeword.sh ./
COPY lang ./lang
RUN chmod +x wyoming-wakeword.sh && ln -s /app/wyoming-wakeword.sh /usr/local/bin/wyoming-wakeword

EXPOSE 10400

ENTRYPOINT ["/app/wyoming-wakeword.sh"]
