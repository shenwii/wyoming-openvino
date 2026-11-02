# wyoming-openvino

[ English ](README.md) | [ 中文 ](README_zh.md) | [ 日本語 ](README_ja.md) | [ 한국어 ](README_ko.md)

---

Wyoming プロトコル + Intel OpenVINO：Intel NPU (Core Ultra)、Arc ディスクリート GPU、内蔵グラフィックス (iGPU)、CPU 向けに最適化された、Home Assistant 連携対応の高性能ウェイクワード検出 (Wake Word)、音声認識 (STT)、音声合成 (TTS) マイクロサービスです。

## 🌟 主な機能

- **ハードウェア完全アクセラレーション & ネイティブ NPU 最適化 (Intel Core Ultra)**:
  - **ウェイクワード (openWakeWord)**: Intel NPU (Meteor Lake / Lunar Lake / Arrow Lake) 向けに徹底最適化。静的形状の自動リシェイプに対応し、超低消費電力での常時待受を実現。`NPU` -> `GPU` -> `CPU` のスムーズなハードウェア自動フォールバックに対応。
  - **音声認識 (STT - Whisper)**: `GPU` (Arc dGPU / iGPU)、`CPU`、`AUTO` スケジューリングに対応。
  - **音声合成 (TTS - Kokoro)**: `CPU` 実行を推奨。適応型スレッドスケジューリングと高い互換性を実現。
- **複数モデル同時検出 & YAML/JSON 設定**:
  - 1 つの音声ストリームで複数のウェイクワード (`Hey Jarvis`、`Alexa`、自作モデル等) を同時にリスニング。
  - YAML または JSON 設定ファイルにより、モデルごとの `cutoff` (しきい値)、`refractory` (不応期)、`device` (実行デバイス) を個別に指定可能。
- **インテリジェント・ストリーミング分割 & 超低遅延生成 (Kokoro TTS)**:
  - 略語や小数を保護 (`Dr.`、`Mr.`、`3.14` 等での不自然な分断を防止)。
  - 文末の句読点 (`。！？!?`) を保持し、自然な抑揚とイントネーションを再現。
  - 節単位のストリーミング出力：最初の節を 100〜300ms 以内に発声開始し、Wyoming TCP 経由で即時配信。
- **加重音声ブレンド (Voice Blending)**:
  - Kokoro-FastAPI スタイルの記法に対応 (例: `zf_xiaobei(0.7)+af_heart(0.3)`、`af_bella(2)+af_sky(1)` 等)。
- **国際化 (i18n) 完全対応**:
  - 英語 (`en`)、中国語 (`zh`)、日本語 (`ja`)、韓国語 (`ko`) の 4 言語に対応。
  - コンソール出力および Home Assistant 側に通知されるサービス情報が環境変数 (`LANG`、`LC_ALL`、`WYOMING_LANG`) に応じて動的に切り替わります。
- **Docker コンテナ対応**:
  - `wyoming-stt`、`wyoming-tts`、`wyoming-wakeword` に分離され、それぞれ独立した `Dockerfile`、`requirements.txt`、シェルスクリプトを同梱。ベースイメージには `docker.io/openvino/ubuntu24_runtime:latest` を採用。

---

## 📁 ディレクトリ構成

```
wyoming-openvino/
├── wyoming-stt/              # 音声認識サービス (Whisper OpenVINO)
│   ├── wyoming-stt.py        # Python サービススクリプト
│   ├── wyoming-stt.sh        # シェルエントリーポイント
│   ├── i18n.py               # 国際化エンジン
│   ├── lang/                 # 言語カタログ (en.json, zh.json, ja.json, ko.json)
│   ├── requirements.txt      # STT 依存パッケージ
│   └── Dockerfile            # コンテナ定義
│
├── wyoming-tts/              # 音声合成サービス (Kokoro OpenVINO)
│   ├── wyoming-tts.py        # Python サービススクリプト
│   ├── wyoming-tts.sh        # シェルエントリーポイント
│   ├── i18n.py               # 国際化エンジン
│   ├── lang/                 # 言語カタログ (en.json, zh.json, ja.json, ko.json)
│   ├── requirements.txt      # TTS 依存パッケージ
│   └── Dockerfile            # マルチステージビルドコンテナ定義
│
├── wyoming-wakeword/         # ウェイクワードサービス (openWakeWord OpenVINO)
│   ├── wyoming-wakeword.py   # Python サービススクリプト
│   ├── wyoming-wakeword.sh   # シェルエントリーポイント
│   ├── wakeword_config.yaml  # 複数モデル設定ファイル
│   ├── i18n.py               # 国際化エンジン
│   ├── lang/                 # 言語カタログ (en.json, zh.json, ja.json, ko.json)
│   ├── requirements.txt      # ウェイクワード依存パッケージ
│   └── Dockerfile            # コンテナ定義
│
├── wakeword_trainer/         # ウェイクワードトレーニングツール
├── README.md                 # 英語ドキュメント (デフォルト)
├── README_zh.md              # 中国語ドキュメント
├── README_ja.md              # 日本語ドキュメント
├── README_ko.md              # 韓国語ドキュメント
└── LICENSE                   # GPL-3.0 ライセンス
```

---

## 📦 インストールと直接実行

### 1. Wyoming STT (音声認識)

```bash
cd wyoming-stt
pip install -r requirements.txt

# GPU アクセラレーションで起動
./wyoming-stt.sh --model-dir /path/to/whisper-openvino --device GPU --port 10020
```

### 2. Wyoming TTS (音声合成)

```bash
cd wyoming-tts
pip install -r requirements.txt

# CPU で起動 (自然な発話ペースとして 0.85 速度推奨)
./wyoming-tts.sh --model-dir /path/to/Kokoro-82M-FP16-OpenVINO --device CPU --speed 0.85 --port 10021
```

### 3. Wyoming Wake Word (ウェイクワード)

```bash
cd wyoming-wakeword
pip install -r requirements.txt

# 方法 A: YAML 設定ファイルで起動 (推奨)
./wyoming-wakeword.sh --config wakeword_config.yaml

# 方法 B: Intel NPU で特定モデルを直接起動
./wyoming-wakeword.sh --model models/hey_jarvis_v0.1.onnx --device NPU --cutoff 0.6 --port 10400
```

---

## 🐳 Docker デプロイ

全サービス共通で `docker.io/openvino/ubuntu24_runtime:latest` をベースイメージとして使用します。

### 1. イメージのビルド

```bash
# STT イメージのビルド
docker build -t wyoming-stt ./wyoming-stt

# TTS イメージのビルド (pyopenjtalk をマルチステージで自動コンパイル)
docker build -t wyoming-tts ./wyoming-tts

# Wake Word イメージのビルド
docker build -t wyoming-wakeword ./wyoming-wakeword
```

### 2. コンテナの起動

#### STT コンテナ (GPU `/dev/dri` マッピング)
```bash
docker run -d \
  --name wyoming-stt \
  --restart unless-stopped \
  --net host \
  --device /dev/dri \
  -v /path/to/whisper-openvino:/models/whisper:ro \
  -e LANG=ja_JP.UTF-8 \
  wyoming-stt \
  --model-dir /models/whisper \
  --device GPU \
  --port 10020
```

#### TTS コンテナ
```bash
docker run -d \
  --name wyoming-tts \
  --restart unless-stopped \
  --net host \
  -v /path/to/Kokoro-82M-FP16-OpenVINO:/models/kokoro:ro \
  -e LANG=ja_JP.UTF-8 \
  wyoming-tts \
  --model-dir /models/kokoro \
  --device CPU \
  --speed 0.85 \
  --port 10021
```

#### Wake Word コンテナ (Intel NPU `/dev/accel` マッピング)
```bash
docker run -d \
  --name wyoming-wakeword \
  --restart unless-stopped \
  --net host \
  --device /dev/accel \
  -v /path/to/models:/app/models \
  -v $(pwd)/wyoming-wakeword/wakeword_config.yaml:/app/wakeword_config.yaml:ro \
  -e LANG=ja_JP.UTF-8 \
  wyoming-wakeword \
  --config /app/wakeword_config.yaml
```

---

## 🌐 国際化 (i18n) 設定

各マイクロサービスは多言語対応モジュールを備え、翻訳テキストは `lang/` ディレクトリ内に分離されています (**英語**: `en.json`、**中国語**: `zh.json`、**日本語**: `ja.json`、**韓国語**: `ko.json`)。新しい言語 (例: ドイツ語 `de.json`) を追加する場合も、Python コードを変更することなく該当ディレクトリに JSON ファイルを追加するだけで対応可能です。

ログおよび Wyoming プロトコルの言語表示は標準の環境変数によって自動的に切り替わります：
- `WYOMING_LANG` (最優先)
- `LC_ALL`
- `LC_MESSAGES`
- `LANG` (例: `ja_JP.UTF-8`, `en_US.UTF-8`, `zh_CN.UTF-8`, `ko_KR.UTF-8`)

---

## 🏠 Home Assistant 連携

Home Assistant で **設定** -> **デバイスとサービス** -> **統合を追加** -> **Wyoming Protocol** を検索：

1. **STT の追加**: ホスト IP、ポート `10020`
2. **TTS の追加**: ホスト IP、ポート `10021`
3. **Wake Word の追加**: ホスト IP、ポート `10400`

追加後、**音声アシスタント (Assist)** から各モデルを選択してご利用ください。

---

## 📜 ライセンス

GNU General Public License v3.0 (GPL v3)
