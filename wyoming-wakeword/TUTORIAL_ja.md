# Wyoming Wake Word (ウェイクワード) クイックスタートチュートリアル

[ English ](TUTORIAL.md) | [ 中文 ](TUTORIAL_zh.md) | [ 日本語 ](TUTORIAL_ja.md) | [ 한국어 ](TUTORIAL_ko.md)

---

このチュートリアルでは、Intel OpenVINO で高速化された **Wyoming Wake Word (openWakeWord)** の Docker コンテナ環境をコピペだけで手軽に立ち上げる手順を解説します。Intel NPU、GPU、CPU のハードウェアアクセラレーションと複数ウェイクワードの並行検知に対応しています。

## 1. モデルと作業ディレクトリの準備

作業ディレクトリにモデル格納用の `models` ディレクトリを作成します：

```bash
mkdir -p models
```

- `wakeword_trainer` で自作・訓練したウェイクワードモデル（`.onnx` ファイル）を `./models` ディレクトリに配置します。
- 公式の事前学習済みモデル（`hey_jarvis_v0.1.onnx`、`alexa_v0.1.onnx` など）を使用する場合、起動時に基礎音響モデル（`melspectrogram.onnx`、`embedding_model.onnx`）が自動ダウンロードされます。

---

## 2. 設定ファイルの取得と編集 (`wakeword_config.yaml`)

日本語コメント付きの設定ファイルを取得します：

```bash
wget https://raw.githubusercontent.com/shenwii/wyoming-openvino/refs/heads/main/wyoming-wakeword/wakeword_config_ja.yaml -O wakeword_config.yaml
```

エディタで `wakeword_config.yaml` を開き、`models` リストに使用するモデルを定義します：

```yaml
models:
  - name: "hey_jarvis"
    phrase: "Hey Jarvis"
    model_path: "models/hey_jarvis_v0.1.onnx"
    cutoff: 0.60
    trigger_level: 2
    sliding_window_size: 3
    trailing_fall: true
    refractory: 2.0
    device: "NPU"
```

---

## 3. `docker-compose.yml` の作成

同じディレクトリに `docker-compose.yml` を作成します：

```yaml
services:
  wyoming-wakeword:
    image: ghcr.io/shenwii/wyoming-wakeword
    container_name: wyoming-wakeword
    ports:
      - "10400:10400"
    devices:
      - /dev/dri/renderD128:/dev/dri/renderD128
      - /dev/accel/accel0:/dev/accel/accel0
    volumes:
      - ./models:/app/models
      - ./wakeword_config.yaml:/app/wakeword_config.yaml
    environment:
      - LANG=ja_JP.UTF-8
    command:
      - --config
      - wakeword_config.yaml
    restart: unless-stopped
```

### 重要なポイントと設定解説
- **コンテナ作業ディレクトリ (`WORKDIR`)**: コンテナの標準作業ディレクトリは `/app` です。`./models:/app/models` と `./wakeword_config.yaml:/app/wakeword_config.yaml` にマウントすることで、パスを相対指定でシンプルに記述できます。
- **デバイスパススルー (`devices`)**:
  - `/dev/accel/accel0:/dev/accel/accel0`: Intel Core Ultra NPU を使用した極小電力の常時監視用。システムに NPU がない場合は行を削除またはコメントアウトしてください。
  - `/dev/dri/renderD128:/dev/dri/renderD128`: Intel 内蔵 GPU や Arc GPU のアクセラレーション用。
  - ハードウェアの自動フォールバック（`NPU` -> `GPU` -> `CPU`）が組み込まれています。

---

## 4. 設定ファイルの重要パラメータ詳細

| パラメータ名 | デフォルト | 役割と調整のポイント |
| :--- | :--- | :--- |
| `cutoff` | `0.70` | **スコア判定しきい値** (0.0〜1.0)。通常は 0.60〜0.80 を指定します。低すぎると誤検知が増え、高すぎると発音を強くする必要があります。 |
| `trigger_level` | `2` | **連続判定フレーム数**。突発ノイズ（テレビの音など）で単一フレームのみ 0.9 を超えるスパイクを無視するため、N 連続フレーム（1フレーム=80ms）基準を満たした時に検知とみなします。 |
| `sliding_window_size` | `3` | **スライディングウィンドウ平均サイズ**。直近 N フレームの移動平均を計算し、偶発的なノイズスパイクを平滑化します（3フレーム ≈ 240ms）。 |
| `trailing_fall` | `true` | **ピーク通過後リリース機構（早期切断防止）**。ウェイクワード後半部分が STT に混入するのを防ぐ最重要機能（詳細は後述）。 |
| `max_trailing_frames` | `8` | **ピーク通過後の最大待機フレーム数**（8フレーム ≈ 640ms）。語尾を長く引き伸ばしたり高ノイズでスコアが下がらない場合のタイムアウト保護機構です。 |
| `trailing_raw_cutoff` | `0.20` | **急降下即時リリーストリガー**。スライディングウィンドウが大きい時の遅延を回避し、最新フレームがこの値未満に急降下した瞬間即時発火します。 |
| `refractory` | `2.0` | **検知後クールダウン時間（秒）**。二重トリガーを防ぐ待機時間です。 |

### 💡 なぜ `trailing_fall`（ピーク通過後リリース）が必要なのか？
ウェイクワードがやや長い場合（約1秒）、ウィンドウ設定次第では前半（約500ms地点）でスコアが `cutoff` を超えてしまうことがあります。
- **この機能がない場合**: 前半の時点で即座にトリガーされ、Home Assistant はウェイクワード監視を終了して STT 録音を開始してしまいます。その結果、ウェイクワードの後半部分がコマンド音声の冒頭に混入し、STT の認識精度が著しく低下します。
- **`trailing_fall: true` の場合**: スコアが基準を超えても即座に切断せず、ピークからの減衰（`0.1 -> 0.7 -> 0.9 -> 0.9 -> 0.7 -> 0.3`）を監視します。発声が終わりスコアがピークから下降したタイミングでトリガーを送信するため、ウェイクワード直後の自然な無音ポーズで綺麗に切り取られ、クリーンな音声が STT に送られます。

---

## 5. サービスの起動と Home Assistant 連携

サービスのバックグラウンド起動：
```bash
docker compose up -d
```

ログ確認：
```bash
docker compose logs -f wyoming-wakeword
```

### Home Assistant への連携
1. Home Assistant の **設定** -> **デバイスとサービス** -> **統合を追加** を開きます。
2. **Wyoming Protocol** を検索し、ホスト IP とポート `10400` を入力します。
3. **音声アシスタント** 設定から、**ウェイクワード** に登録されたモデルを指定します。
