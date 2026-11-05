# Wyoming STT (音声認識) クイックスタートチュートリアル

[ English ](TUTORIAL.md) | [ 中文 ](TUTORIAL_zh.md) | [ 日本語 ](TUTORIAL_ja.md) | [ 한국어 ](TUTORIAL_ko.md)

---

このチュートリアルでは、Intel OpenVINO で高速化された **Wyoming STT (Whisper)** の Docker コンテナ環境をコピペだけで手軽に立ち上げる手順を解説します。

## 1. モデルのダウンロード

Whisper モデルは OpenVINO IR（Intermediate Representation）形式である必要があります。本例では認識精度とリソース消費のバランスに優れた **Whisper Medium FP16** モデルを使用します。

> 💡 **他のサイズや量子化モデル**: リソースを節約したい場合（`base`, `small` など）や高精度を求める場合（`large-v3-turbo` など）は、Hugging Face の [OpenVINO 公式組織ページ](https://huggingface.co/OpenVINO) からモデル ID を取得して差し替えてください。

作業ディレクトリで以下のコマンドを実行し、`models` ディレクトリを作成してモデルをダウンロードします：

```bash
mkdir -p models
docker run --rm -v ./models:/models docker.io/library/python:3.11 bash -c '
pip install -U huggingface_hub
hf download OpenVINO/whisper-medium-fp16-ov --local-dir /models/whisper-medium-fp16-ov
'
```

---

## 2. `docker-compose.yml` の作成

同じディレクトリに `docker-compose.yml` を作成します：

```yaml
services:
  wyoming-stt:
    image: ghcr.io/shenwii/wyoming-stt
    container_name: wyoming-stt
    ports:
      - "10020:10020"
    devices:
      - /dev/dri/renderD128:/dev/dri/renderD128
    volumes:
      - ./models:/models
    environment:
      - LANG=ja_JP.UTF-8
    command:
      - --model-dir
      - /models/whisper-medium-fp16-ov
      - --device
      - GPU
      - --port
      - "10020"
    restart: unless-stopped
```

### 重要なポイントと設定解説
- **ポート番号**: STT の標準ポートは `10020` です（TTS は `10021`）。
- **ハードウェアアクセラレーション (`devices`)**:
  - `/dev/dri/renderD128:/dev/dri/renderD128`: Intel 内蔵 GPU (iGPU) または Arc ディスクリート GPU をコンテナにパススルーして GPU 推論を行います。
  - CPU のみで実行する場合は `devices` 行を削除し、command 内の `--device` を `CPU` に変更してください。
- **言語設定**:
  - デフォルトでは自動言語判別（`--language auto`）が有効です。日本語のみに固定したい場合は `--language ja` を追加できます。
- **環境変数 (`LANG=ja_JP.UTF-8`)**: コンソールログや Home Assistant 向け Wyoming メタデータ表示を日本語に切り替えます。

---

## 3. サービスの起動

Docker Compose でバックグラウンド起動します：

```bash
docker compose up -d
```

ログを確認し、モデルが正しくロードされたことを確認します：

```bash
docker compose logs -f wyoming-stt
```

---

## 4. Home Assistant への連携設定

1. Home Assistant の管理画面から **設定** -> **デバイスとサービス** -> **統合を追加** を開きます。
2. **Wyoming Protocol** を検索して選択します。
3. STT サービスを動かしているマシンの IP アドレスとポート `10020` を入力して送信します。
4. **音声アシスタント (Voice Assistants)** 設定を開き、パイプラインの **音声テキスト変換 (Speech-to-text)** で `wyoming-openvino-stt` を選択します。
