# Wyoming / openWakeWord NPU ウェイクワード学習スイート

**[openWakeWord](https://github.com/dscripka/openWakeWord)** および **[MicroWakeWordV2Trainer](https://github.com/JohnnyPrimus/MicroWakeWordV2Trainer)** をベースに構築された、**Intel NPU (OpenVINO VPUX / Core Ultra)** 向けエンドツーエンド学習スイートです。

---

## 🌟 主な特徴

1. **統一された設定ファイル (`config.yaml`)**:
   - `wakeword` セクションに設定を集約: ターゲットフレーズ、`friend_name`（空白なしのモデル識別名）、エイリアス、類似妨害語（Hard Negatives）。
   - 多言語対応（`en`、`zh`、`ja` など）。
2. **デュアルエンジン TTS 合成 (`generate_tts.py`)**:
   - **ローカルエンジン (Kokoro OpenVINO TTS)**: 高速オフライン音声合成、話者ブレンド対応。
   - **クラウドエンジン (Microsoft Edge-TTS)**: 高品質ニューラル音声。
3. **現実的な音響シミュレーション (`augment_data.py`)**:
   - MIT Environmental RIR を使用した室内残響シミュレーション。
   - 生活環境ノイズのランダム SNR（5〜25dB）重畳。
4. **Intel NPU に特化した最適化 (`train.py`)**:
   - **完全静的形状**: `[1, 16, 96] -> [1, 1]` の静的 ONNX モデルを出力。
   - **純粋な演算子グラフ**: `Gemm + ReLU + Sigmoid` のみを使用し、100% NPU ハードウェアアクセラレーションを実現。
   - **ハードウェア加速**: CPU、CUDA、および Intel GPU（PyTorch 2.4+ `xpu` / IPEX）をサポート。

---

## 📁 ディレクトリ構成

```text
wakeword_trainer/
├── config.sample.yaml     # 設定例 (英語デフォルト)
├── config.sample_zh.yaml  # 設定例 (中国語)
├── config.sample_ja.yaml  # 設定例 (日本語)
├── config.sample_ko.yaml  # 設定例 (韓国語)
├── requirements.txt       # 依存パッケージ
├── i18n.py                # 国際化モジュール
├── lang/                  # 多言語定義 (en, zh, ja, ko)
├── setup_libs.py          # 特徴抽出基本モデルダウンロード
├── download_data.py       # RIR・ノイズデータセットダウンロード
├── generate_tts.py        # TTS 音声データ合成
├── augment_data.py        # 音響拡張シミュレーション
├── train.py               # DNN 学習および NPU 最適化 ONNX 出力
├── run_pipeline.py        # パイプライン自動実行スクリプト
└── README.md              # 説明ドキュメント
```

---

## 🛠️ インストール

```bash
cd wakeword_trainer
pip install -r requirements.txt
```

---

## ⚙️ 設定方法 (`config.yaml`)

設定例をコピーして使用します:
```bash
cp config.sample_ja.yaml config.yaml
```

`wakeword` セクションを編集するだけで設定完了です:
```yaml
wakeword:
  phrase: "hey jarvis"            # ターゲットウェイクワード
  friend_name: "hey_jarvis_v1"    # モデル識別名 ({friend_name}.onnx として出力)
  candidate_phrases:              # 候補フレーズ
    - "jarvis jarvis"
  hard_negatives:                 # 誤検出防止用ハード負例
    - "hey harvest"
    - "電気をつけて"
  language: "en"                  # 言語コード
```

---

## 🚀 実行方法

パイプラインの一括実行:
```bash
python run_pipeline.py --config config.yaml
```

クイックテストモード:
```bash
python run_pipeline.py --quick-mode
```

---

## 🎯 Wyoming サービスとの連携

```bash
cd ../wyoming-wakeword
python wyoming-wakeword.py \
    --model ../wakeword_trainer/output/hey_jarvis_v1.onnx \
    --device NPU \
    --cutoff 0.55 \
    --refractory 2.0
```
