# ウェイクワードモデル学習パイプライン (wakeword_trainer) クイックスタートチュートリアル

[ English ](TUTORIAL.md) | [ 中文 ](TUTORIAL_zh.md) | [ 日本語 ](TUTORIAL_ja.md) | [ 한국어 ](TUTORIAL_ko.md)

---

このチュートリアルでは、Docker を利用して**独自のカスタムウェイクワードをエンドツーエンドで全自動学習**する手順を解説します。
Kokoro OpenVINO と Microsoft Edge-TTS のデュアルエンジンによる音声データ自動合成、実部屋の反射音シミュレーション（MIT RIR）、環境雑音の重畳を行い、**Intel NPU 向けに最適化された ONNX モデル** と **ESPHome / microWakeWord 互換の TFLite モデル** をワンクリックで学習・出力します。

---

## 1. リポジトリのクローン

```bash
git clone --depth=1 'https://github.com/shenwii/wyoming-openvino'
cd wyoming-openvino
```

---

## 2. 設定ファイルのコピーと編集

日本語向けサンプル設定ファイルをコピーします：

```bash
cp wakeword_trainer/config.sample_ja.yaml ./config.yaml
```

エディタで `./config.yaml` を開き、先頭の `wakeword` 項目をお好みの単語に変更します：

```yaml
wakeword:
  phrase: "こんにちは"          # 目標とするウェイクワード
  friend_name: "konnichiwa_v1"  # モデル識別名 ({friend_name}.onnx および {friend_name}.tflite として出力)
  candidate_phrases:            # 類似・連読バリエーション
    - "こんにちはこんにちは"
    - "はいこんにちは"
  hard_negatives:               # 誤検知防止用の近音・日常フレーズ（重要）
    - "こんばんは"
    - "さようなら"
    - "でんきをつけて"
    - "でんきをけして"
  language: "ja"                # 言語コード (ja, en, zh, ko など)
```

> 💡 **推奨学習デバイス**: `training.device` は `"cpu"` のままにしておくことを推奨します（高速かつドライバ競合がありません）。

---

## 3. Docker によるワンクリック自動学習の開始

以下のコマンドを実行すると、隔離された Docker 環境内で依存関係のセットアップ、Kokoro ローカルオフライン音声モデルの取得、学習パイプラインの実行が全自動で行われます：

```bash
docker run --rm \
  --name wakeword-trainer \
  --device /dev/dri/renderD128:/dev/dri/renderD128 \
  -v $(pwd):/app \
  -v $(pwd)/config.yaml:/app/wakeword_trainer/config.yaml \
  -d docker.io/library/python:3.11 bash -c '
cd /app/wakeword_trainer
apt update && apt install -y ffmpeg
pip install --upgrade pip
pip install -U huggingface_hub
hf download OpenVINO/Kokoro-82M-int8-ov --local-dir ../Kokoro-82M-int8-ov
pip install -r requirements.txt
pip install -r ../wyoming-tts/requirements.txt
python run_pipeline.py --config config.yaml
'
```

> 💡 **ポイント**:
> - Intel GPU を搭載していない環境の場合は、`--device /dev/dri/renderD128:/dev/dri/renderD128` の行を削除してください。
> - コンテナはバックグラウンドで安全に実行されます。

---

## 4. 学習進捗の確認

リアルタイムで実行ログを確認できます：

```bash
docker logs -f wakeword-trainer
```

以下のパイプラインが自動的に進行します：
1. **基礎音響モデルとデータセットの取得**（openWakeWord 基礎特徴量モデル、MIT 部屋インパルス応答、環境ノイズなど）。
2. **デュアルエンジン音声合成**（Kokoro TTS と Edge-TTS による数千件の正例・負例データの多様化生成）。
3. **室内音響シミュレーションとノイズ重畳**（反響や実生活ノイズのミキシング）。
4. **ニューラルネットワーク学習と全フォーマット出力**。

---

## 5. 生成モデルの利用方法

学習が完了すると、成果物は `./wakeword_trainer/output/` ディレクトリに出力されます：

- **`{friend_name}.onnx`**:
  Intel NPU / GPU / CPU 向け静的形状 ONNX モデル。
  👉 **利用方法**: `wyoming-wakeword/models/` にコピーし、`wakeword_config.yaml` に追記して Home Assistant の常時監視に利用します。
- **`{friend_name}.tflite`**:
  標準 32-bit 浮動小数点 TensorFlow Lite モデル。
  👉 **利用方法**: ESP32-S3 等を用いた **ESPHome microWakeWord** スマートマイク端末に配置します。
- **`{friend_name}.json`**:
  推奨 `cutoff` 判定しきい値を含むメタデータ。
- **`{friend_name}.pt`**:
  PyTorch 重みファイル（後から再エクスポートする際に使用可能）。
