# Wyoming TTS (音声合成) クイックスタートチュートリアル

[ English ](TUTORIAL.md) | [ 中文 ](TUTORIAL_zh.md) | [ 日本語 ](TUTORIAL_ja.md) | [ 한국어 ](TUTORIAL_ko.md)

---

このチュートリアルでは、Intel OpenVINO で高速化された **Wyoming TTS (Kokoro-82M)** の Docker コンテナ環境をコピペだけで手軽に立ち上げる手順を解説します。

## 1. モデルのダウンロード

本例では省メモリかつ高音質な **Kokoro-82M INT8 量子化モデル** を使用します。

> 💡 **モデルオプション**: 量子化されていないネイティブ FP16 版を利用したい場合は、`Echo9Zulu/Kokoro-82M-FP16-OpenVINO` に差し替えることも可能です。

作業ディレクトリで以下のコマンドを実行し、`models` ディレクトリを作成してモデルをダウンロードします：

```bash
mkdir -p models
docker run --rm -v ./models:/models docker.io/library/python:3.11 bash -c '
pip install -U huggingface_hub
hf download OpenVINO/Kokoro-82M-int8-ov --local-dir /models/Kokoro-82M-int8-ov
'
```

---

## 2. `docker-compose.yml` の作成

同じディレクトリに `docker-compose.yml` を作成します：

```yaml
services:
  wyoming-tts:
    image: ghcr.io/shenwii/wyoming-tts
    container_name: wyoming-tts
    ports:
      - "10021:10021"
    volumes:
      - ./models:/models
    environment:
      - LANG=ja_JP.UTF-8
    command:
      - --model-dir
      - /models/Kokoro-82M-int8-ov
      - --device
      - CPU
      - --port
      - "10021"
      - --speed
      - "0.85"
    restart: unless-stopped
```

### 重要なポイントと設定解説
- **ポート番号**: TTS の標準ポートは `10021` です（STT は `10020`）。
- **モデルディレクトリの一致**: ダウンロードした保存先 `/models/Kokoro-82M-int8-ov` と、command に指定する `--model-dir` のパスが完全に一致している必要があります。
- **推奨実行デバイス (`--device CPU`)**:
  - Kokoro TTS は **CPU** 上で動作させることを強く推奨します。OpenVINO の CPU 命令セット最適化（AVX2/AVX-512/AMX）により、GPU シェーダーコンパイルの引っかかりがなく、100〜300ms 以内で最初の音声チャンクが即時再生されます。
  - GPU で実行したい場合は `devices: - /dev/dri/renderD128:/dev/dri/renderD128` をマウントし、`--device` を `GPU` に変更してください。
- **読み上げ速度 (`--speed 0.85`)**: Kokoro の基準速度はやや速めのため、`0.85` を指定すると最も聞き取りやすく自然なトーンになります。
- **話者（ボイス）の指定と重み付けブレンド (`--voice`)**:
  - 日本語ボイス例: `jf_alpha`, `jf_gongitsune`, `jf_tebaf` (女性)、`jm_kumo` (男性)。
  - 複数話者のブレンド指定も可能です（例: `- --voice` と `- jf_alpha(0.7)+af_heart(0.3)`）。

---

## 3. サービスの起動

Docker Compose でバックグラウンド起動します：

```bash
docker compose up -d
```

ログを確認し、モデルが正しくロードされたことを確認します：

```bash
docker compose logs -f wyoming-tts
```

---

## 4. Home Assistant への連携設定

1. Home Assistant の管理画面から **設定** -> **デバイスとサービス** -> **統合を追加** を開きます。
2. **Wyoming Protocol** を検索して選択します。
3. TTS サービスを動かしているマシンの IP アドレスとポート `10021` を入力して送信します。
4. **音声アシスタント (Voice Assistants)** 設定を開き、パイプラインの **テキスト読み上げ (Text-to-speech)** で `kokoro-openvino` を選択します。
