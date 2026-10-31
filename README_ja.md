# wyoming-openvino

[ 中文 ](README.md) | [ English ](README_en.md) | [ 日本語 ](README_ja.md)

---

Wyoming プロトコル + Intel OpenVINO：Intel 独立型 GPU（Arc）、内蔵 GPU（iGPU）、および CPU 上で音声認識（STT）および音声合成（TTS）サービスを高速に実行し、Home Assistant とシームレスに連携します。

## 🌟 主な機能

- **公式 OpenVINO モデル形式を直接サポート**：
  - **TTS (Kokoro)**：Hugging Face からダウンロードした公式 OpenVINO ディレクトリ（`openvino_model.xml`、`openvino_model.bin`、および `voices/*.pt` や `voices/*.bin`）をそのまま指定可能。追加の変換作業は不要です。
  - **STT (Whisper)**：Optimum/OpenVINO 形式でエクスポートされた Whisper モデルを直接サポート。
- **シンプルなコマンドラインインターフェース（`--model-dir` で統一）**：
  - STT と TTS の両方で、モデルディレクトリを指定するだけでモデル構造・重み・利用可能な全話者音声を自動的に認識・読み込みます。
- **多言語自動判別 & Wyoming プロトコルによる動的切り替え**：
  - コマンドラインの `--language` はデフォルトで `auto`（省略可能）。
  - **STT**：Home Assistant からのリクエストで指定された言語を優先。未指定の場合は Whisper の自動言語検出機能（日本語・英語・中国語など 90 以上の言語）が働きます。
  - **TTS**：リクエストで指定された話者プレフィックス（`zf_*` -> 中国語、`jf_*` -> 日本語、`af_*` -> 英語）やテキスト文字（ひらがな/カタカナ/漢字/アルファベット）から最適な G2P エンジンへ自動ルーティング。
- **Intel ハードウェアアクセラレーション**：
  - `GPU`（Intel Arc dGPU / iGPU）、`CPU`、`AUTO`、および `MULTI:CPU,GPU` ハイブリッド推論に対応。

---

## 📦 環境要件と依存関係のインストール

### 1. システム要件
- Python >= 3.10
- Intel GPU ドライバ & OpenVINO Runtime

### 2. 依存関係のインストール

デプロイ用途に応じて、必要な依存関係ファイルを選択してインストールします：

```bash
# パターン A: STT (音声認識) のみ実行（軽量、OpenVINO GenAI & Wyoming のみ）
pip install -r requirements-stt.txt

# パターン B: TTS (音声合成) のみ実行（Kokoro、音声ファイル解析、および日/英/中 G2P 依存関係を含む）
pip install -r requirements-tts.txt

# パターン C: STT と TTS を両方実行
pip install -r requirements.txt
```

---

## 🧠 モデルの準備

### 1. STT モデル (Whisper)

標準的な OpenVINO Whisper モデルディレクトリ構成：
```
whisper-openvino/
├── openvino_model.xml
├── openvino_model.bin
├── config.json
├── tokenizer.json
└── ...
```

### 2. TTS モデル (Kokoro)

Hugging Face（例：`Kokoro-82M-FP16-OpenVINO`）からダウンロードした標準ディレクトリ構成：
```
Kokoro-82M-FP16-OpenVINO/
├── openvino_model.xml     # モデルネットワーク定義
├── openvino_model.bin     # モデルバイナリ重み
├── config.json            # モデル設定
├── openvino_config.json
├── voices/                # 話者音声ベクトルフォルダ
│   ├── af_heart.pt        # 英語（アメリカ）女性
│   ├── zf_xiaobei.pt      # 中国語女性
│   ├── jf_teio.pt         # 日本語音声
│   └── ...
└── ...
```
> [!NOTE]
> `voices/` フォルダ内の各ファイルは個別の話者音声を表します。サービス起動時に全話者ファイルを自動スキャンし、Home Assistant に自動登録します。

---

## 🚀 クイックスタート

### 1. STT (音声認識) サービスの起動

```bash
python wyoming-stt.py \
    --model-dir /path/to/whisper-openvino \
    --device GPU \
    --port 10020
```

### 2. TTS (音声合成) サービスの起動

```bash
python wyoming-tts.py \
    --model-dir /path/to/Kokoro-82M-FP16-OpenVINO \
    --device GPU \
    --port 10021
```

---

## ⚙️ コマンドライン引数一覧

### STT 引数 (`wyoming-stt.py`)

| 引数 | 短縮 | デフォルト | 説明 |
|---|---|---|---|
| `--model-dir` | `-m` | (必須) | Whisper モデルディレクトリのパス |
| `--device` | `-d` | `GPU` | 推論デバイス (`CPU` / `GPU` / `AUTO` / `MULTI:CPU,GPU`) |
| `--language` | `-l` | `auto` | デフォルト言語（HA のリクエスト優先、未指定時は自動検出） |
| `--port` | `-p` | `10020` | リッスンポート番号 |
| `--host` | | `0.0.0.0` | リッスンアドレス |

### TTS 引数 (`wyoming-tts.py`)

| 引数 | 短縮 | デフォルト | 説明 |
|---|---|---|---|
| `--model-dir` | `-m` | (必須) | Kokoro モデルディレクトリ（xml/bin および voices を自動検索） |
| `--device` | `-d` | `GPU` | 推論デバイス (`CPU` / `GPU` / `AUTO` / `MULTI:CPU,GPU`) |
| `--language` | `-l` | `auto` | デフォルト言語（HA のリクエスト優先、未指定時は自動推測） |
| `--speed` | `-s` | `1.0` | 発話速度倍率 |
| `--port` | `-p` | `10021` | リッスンポート番号 |

---

## 🏠 Home Assistant との連携設定

Home Assistant の管理画面から **設定** -> **デバイスとサービス** -> **統合を追加** -> **Wyoming Protocol** を検索します：

1. STT の追加：ホスト IP を入力、ポート番号に `10020` を指定
2. TTS の追加：ホスト IP を入力、ポート番号に `10021` を指定

追加後、Home Assistant の「音声アシスタント (Assist)」設定画面にて、言語や好みの Kokoro 話者を自由に選択できます。

## 📜 ライセンス

GNU General Public License v3.0 (GPL v3)
