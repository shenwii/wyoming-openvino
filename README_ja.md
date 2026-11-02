# wyoming-openvino

[ 中文 ](README.md) | [ English ](README_en.md) | [ 日本語 ](README_ja.md)

---

Wyoming プロトコル + Intel OpenVINO：Intel NPU（Core Ultra）、独立型 GPU（Arc）、内蔵 GPU（iGPU）、および CPU 上でウェイクワード検出（Wake Word）、音声認識（STT）、および音声合成（TTS）サービスを高速に実行し、Home Assistant とシームレスに連携します。

## 🌟 主な機能

- **ハードウェア完全アクセラレーション & Intel NPU 最適化（Core Ultra）**：
  - **Wake Word (openWakeWord)**：Intel NPU（Meteor Lake / Lunar Lake / Arrow Lake）向けに静的形状変形（Static Shape Reshape）を深く最適化。超低消費電力・常時稼働のウェイクワード検出を実現し、`NPU` -> `GPU` -> `CPU` のスムーズな自動フォールバックに対応。
  - **STT (Whisper)**：`GPU`（Intel Arc dGPU / iGPU）、`CPU`、および `AUTO` ハイブリッド推論に対応。
  - **TTS (Kokoro)**：`CPU` 推論を推奨、CPU トポロジーに応じた動的スレッド最適化。
- **マルチウェイクワード同時検出 & YAML/JSON 柔軟設定**：
  - 単一のオーディオストリームで複数のウェイクワード（`Hey Jarvis`、`Alexa`、自作モデルなど）を並行してリッスン可能。特徴抽出パイプラインを共有し、低負荷を実現。
  - YAML または JSON 設定ファイルに対応：モデルごとに `cutoff`（閾値）、`refractory`（クールダウン時間）、`device`（実行デバイス）を個別に上書き設定可能。
- **公式 OpenVINO モデル形式を直接サポート**：
  - **TTS (Kokoro)**：Hugging Face からダウンロードした公式 OpenVINO ディレクトリ（`openvino_model.xml`、`openvino_model.bin`、および `voices/*.pt` や `voices/*.bin`）をそのまま指定可能。追加の変換作業は不要です。
  - **STT (Whisper)**：Optimum/OpenVINO 形式でエクスポートされた Whisper モデルを直接サポート。
  - **Wake Word**：openWakeWord 形式の `.onnx`、`.tflite`、および OpenVINO 原生 `.xml` モデルを直接サポート。
- **シンプルなコマンドラインインターフェース（`--model-dir` や `--config` で統一）**：
  - 各サービスで極めてシンプルな起動コマンドとモデル自動検出をサポート。
- **スマート文分割＆低遅延ストリーミング発話（Kokoro-FastAPI の最適化を導入）**：
  - 略語および数値の保護：`Dr.`、`Mr.`、`etc.`、`e.g.` や浮動小数点数（`3.14` など）での誤った途中改行・切断を防止。
  - 末尾音響記号（`。！？!?`）の保持：文末記号を維持して自然な抑揚・語調起伏・ポーズを実現。
  - 文単位のストリーミング送信：Wyoming オーディオチャンクにより、最初の文が約1秒で即時発声開始。長文でも10数秒待つ必要がありません。
- **重み付き複数話者音声ブレンド（Voice Blending）**：
  - Kokoro-FastAPI 互換の音声混合構文（`zf_xiaobei(0.7)+af_heart(0.3)`、`af_bella(2)+af_sky(1)`、`zf_xiaobei+af_heart` など）に対応し、重みを自動正規化して話者ベクトルを線形合成。
- **多言語自動判別 & Wyoming プロトコルによる動的切り替え**：
  - コマンドラインの `--language` はデフォルトで `auto`（省略可能）。
  - **STT**：Home Assistant からのリクエストで指定された言語を優先。未指定の場合は Whisper の自動言語検出機能（日本語・英語・中国語など 90 以上の言語）が働きます。
  - **TTS**：リクエストで指定された話者プレフィックス（`zf_*` -> 中国語、`jf_*` -> 日本語、`af_*` -> 英語）やテキスト文字（ひらがな/カタカナ/漢字/アルファベット）から最適な G2P エンジンへ自動ルーティング。中英混在文は `en_callable` と `EspeakFallback` で滑らかに処理し、純英文節はネイティブ英語音素へ自動切り替え。

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

# パターン C: Wake Word (ウェイクワード) のみ実行（OpenVINO、Wyoming、PyYAML）
pip install -r requirements-wakeword.txt

# パターン D: 全機能インストール (STT + TTS + Wake Word)
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

### 3. Wake Word ウェイクワードモデル (openWakeWord)

`models/` フォルダ内に配置：
```
models/
├── melspectrogram.onnx      # 特徴抽出バックボーン（未検出時は公式 Release より自動ダウンロード）
├── embedding_model.onnx     # 特徴抽出バックボーン（未検出時は公式 Release より自動ダウンロード）
├── hey_jarvis_v0.1.onnx     # 訓練済みまたは公式事前学習モデル
├── alexa_v0.1.onnx
└── ...
```
> [!NOTE]
> `.onnx`、`.tflite`、および OpenVINO 原生 `.xml` 形式をサポート。Intel Core Ultra NPU 向けに静的形状変形を自動実行し、GPU/CPU へのフォールバックにも対応します。

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
    --device CPU \
    --speed 0.85 \
    --port 10021
```

> [!TIP]
> **推奨発話速度**：Kokoro のデフォルト速度 `1.0` は日本語や中国語では少し早口に聞こえる傾向があります。起動時に `--speed 0.85`（または `0.8`）を指定すると、非常に自然で落ち着いた聞き取りやすい発話になります。
> **推奨推論デバイス**：Kokoro-82M-FP16 はモデル内演算子の互換性および高速な応答性を確保するため、`CPU` デバイスでの実行を推奨します。

### 3. Wake Word (ウェイクワード) サービスの起動

**方法 A: YAML 設定ファイルを使用（複数モデル並行検出に推奨）**
```bash
python wyoming-wakeword.py --config wakeword_config.yaml
```

**方法 B: コマンドラインでモデルと NPU を直接指定**
```bash
python wyoming-wakeword.py \
    --model models/hey_jarvis_v0.1.onnx \
    --device NPU \
    --cutoff 0.6 \
    --port 10400
```

**方法 C: 指定ディレクトリ内の全モデルを自動読み込み**
```bash
python wyoming-wakeword.py --model-dir ./models --device NPU
```

### 4. スタンドアロン動作確認ツール

#### A. TTS 音声合成と WAV 保存 (`test_tts.py`)
```bash
# 基本的な日本語・中国語テスト
python test_tts.py "我来测试一下语音合成的效果。" --model-dir /path/to/Kokoro-82M-FP16-OpenVINO -o test.wav

# 音声ブレンドテスト（中国語女性 70% + 英語女性 30% の比率で合成）
python test_tts.py "今天天气真好，我们来测试 OpenVINO 语音合成。" \
    --model-dir /path/to/Kokoro-82M-FP16-OpenVINO \
    -v "zf_xiaobei(0.7)+af_heart(0.3)" \
    --speed 0.85 \
    -o blend.wav
```

#### B. Wake Word 検出テスト (`test_wakeword.py`)
```bash
# パターン 1: ローカルの OpenVINO で直接 WAV 音声を評価
python test_wakeword.py sample.wav --config wakeword_config.yaml --device NPU

# パターン 2: クライアントとして稼働中の Wyoming ウェイクワードサービスに接続
python test_wakeword.py sample.wav --server 127.0.0.1:10400
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
| `--device` | `-d` | `CPU` | 推論デバイス (`CPU` 推奨 / `GPU` / `AUTO`) |
| `--language` | `-l` | `auto` | デフォルト言語（HA のリクエスト優先、未指定時は自動推測） |
| `--speed` | `-s` | `1.0` | 発話速度倍率（自然な速度には `0.85` または `0.8` を推奨） |
| `--port` | `-p` | `10021` | リッスンポート番号 |

### Wake Word 引数 (`wyoming-wakeword.py`)

| 引数 | 短縮 | デフォルト | 説明 |
|---|---|---|---|
| `--config` | `-c` | `None` | YAML または JSON 設定ファイルパス（複数モデル・個別閾値の設定に対応） |
| `--model` | `-m` | `None` | ウェイクワードモデルパス（複数指定可能） |
| `--model-dir` | | `None` | 指定ディレクトリ配下の全モデルを自動検出・読み込み |
| `--device` | `-d` | `NPU` | 推論デバイス（`NPU` 優先 / `GPU` / `CPU` / `AUTO`、自動フォールバック機能付き） |
| `--cutoff` | | `0.5` | デフォルト閾値 (0.0 ~ 1.0) |
| `--refractory` | | `2.0` | 検出後のクールダウン時間 (秒) |
| `--port` | `-p` | `10400` | リッスンポート (HA Wyoming openWakeWord 公式ポート: 10400) |
| `--host` | | `0.0.0.0` | リッスンアドレス |
| `--log-level` | | `INFO` | ログ出力レベル (`DEBUG` / `INFO` / `WARNING` / `ERROR`) |
| `--melspec-model` | | `None` | MelSpectrogram 特徴抽出モデルパス（任意） |
| `--embedding-model` | | `None` | Embedding 特徴抽出モデルパス（任意） |

---

## 🏠 Home Assistant との連携設定

Home Assistant の管理画面から **設定** -> **デバイスとサービス** -> **統合を追加** -> **Wyoming Protocol** を検索します：

1. STT の追加：ホスト IP を入力、ポート番号に `10020` を指定
2. TTS の追加：ホスト IP を入力、ポート番号に `10021` を指定
3. Wake Word の追加：ホスト IP を入力、ポート番号に `10400` を指定

追加後、Home Assistant の「音声アシスタント (Assist)」設定画面にて、言語や好みの Kokoro 話者を自由に選択でき、超低消費電力・高感度なウェイクワード機能を利用できます。

## 📜 ライセンス

GNU General Public License v3.0 (GPL v3)
