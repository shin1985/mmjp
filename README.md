# MMJP

MMJP は **日本語テキストをサブワード（もしくは単語相当）に分割**するための、超軽量な C99 実装です。
推論時は **(1) 文字種ベース CRF** と **(2) Unigram LM（辞書）** のスコアを足し合わせたラティス上で Viterbi を回して分割します。

- C99 / 依存ライブラリなし（標準 C のみ）
- 推論時 `malloc` なし・整数スコア（Q8.8）中心
- 学習ツール（モデル生成）とトークナイザ（推論）を同梱
- **Lossless Tokenization**（可逆トークナイズ）: 空白・改行を完全保存
- **Subword Regularization（確率的分割）対応**（FFBS / N-best）
- **言語非依存の文字種分類**（ASCII / UTF8LEN / RANGES モード対応）
- **Python バインディング** 同梱（`pip install .` でローカルビルド）

> この README は「コードの説明」です。配布する **model.bin のライセンスは学習コーパスに依存**します（後述）。

---

## クイックスタート

### 1. ビルド（C ツール）

Linux/macOS の例（GCC/Clang）。

```bash
cd tools

# 学習ツール (mmjp_train)
gcc -O3 -std=c99 \
  -I.. -I../double_array -I../npycrf_lite -I../suffix_array -I../unilm_mdl \
  -o mmjp_train mmjp_train.c mmjp_model.c \
  ../suffix_array/sa_utf8.c ../unilm_mdl/unilm_mdl.c \
  ../double_array/double_array_trie.c ../npycrf_lite/npycrf_lite.c \
  ../mmjp_lossless.c -lm

# 推論ツール (mmjp_tokenize)
gcc -O3 -std=c99 \
  -I.. -I../double_array -I../npycrf_lite \
  -o mmjp_tokenize mmjp_tokenize.c mmjp_model.c \
  ../double_array/double_array_trie.c ../npycrf_lite/npycrf_lite.c \
  ../mmjp_lossless.c -lm
```

### 2. 推論（同梱モデルで即実行）

```bash
echo '東京都に住んでいます。' | ./tools/mmjp_tokenize --model models/mmjp_wiki.bin
```

### 3. 学習（独自モデル生成）

```bash
./tools/mmjp_train \
  --corpus your_corpus.txt \
  --out model.bin \
  --vocab 8000 \
  --max_piece_len 8
```

---

## 同梱モデル

`models/mmjp_wiki.bin`（275KB）を同梱しています。

| 項目 | 値 |
|------|-----|
| 学習コーパス | Wikipedia 日本語版 20k 文 |
| 語彙サイズ | 8,000 |
| CRF 学習 | Janome 銀ラベル 800 文で教師あり最適化 |
| lambda0 | 0.2（バランス設定） |
| 境界 F1 | 0.836 |

```bash
# 基本的なトークナイズ
echo '東京都に住んでいます。' | ./tools/mmjp_tokenize --model models/mmjp_wiki.bin

# Python から使う
import mmjp
m = mmjp.Model("models/mmjp_wiki.bin")
print(m.tokenize("東京都に住んでいます。"))
```

> **注意**: 同梱モデルは Wikipedia（CC BY-SA）で学習しています。

---

## ベンチマーク

Wikipedia コーパス + Janome 分割を銀ラベルとした評価結果です。

### 評価条件

- 学習: Wikipedia 文 20k（教師なし） + Janome 銀ラベル 800 文（教師あり CRF）
- 評価: Janome 銀ラベル 181 文（単語開始境界 F1）
- vocab=8000, max_piece_len=8, CRF: L-BFGS epochs=30 l2=1e-4

### 結果（lambda0 別）

`lambda0` は CRF と LM の混合比率です（小さいほど CRF 寄り＝細かい分割）。

| lambda0 | F1 | Precision | Recall | tokens/sent | token len | throughput |
|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | **0.839** | 0.851 | 0.827 | 29.3 | 1.82 | 50k sent/s |
| 0.15 | 0.838 | 0.875 | 0.804 | 27.7 | 1.92 | 50k sent/s |
| 0.20 | 0.836 | 0.891 | 0.787 | 26.6 | 2.00 | 50k sent/s |
| 0.30 | 0.825 | 0.930 | 0.740 | 24.0 | 2.22 | 50k sent/s |
| 0.50 | 0.778 | 0.962 | 0.653 | 20.5 | 2.60 | 50k sent/s |

### 推奨設定

- **最大 F1**: `lambda0=0.05`（F1=0.839、細かい分割）
- **バランス**: `lambda0=0.2`（F1=0.836、平均トークン長 2.0 文字）← 同梱モデル

---

## 機能詳細

### Lossless Tokenization（可逆トークナイズ）

空白・タブ・改行を特殊文字（▁▂▃▄）に置き換えて完全に保存します。
元テキストへの完全なラウンドトリップが可能です。

#### 学習時

```bash
./tools/mmjp_train \
  --corpus your_corpus.txt \
  --out model.bin \
  --lossless_ws 1 \
  --lossless_eol 1
```

#### 推論時

```bash
# トークナイズ
cat input.txt | ./tools/mmjp_tokenize --model model.bin --lossless_ws -1 --read_all 1 > tokens.txt

# デトークナイズ（完全復元）
cat tokens.txt | ./tools/mmjp_tokenize --model model.bin --detok --lossless_ws -1 > restored.txt

# input.txt と restored.txt は同一になります
```

#### メタ文字

| 元文字 | メタ文字 | Unicode |
|--------|----------|---------|
| スペース | ▁ | U+2581 |
| タブ | ▂ | U+2582 |
| LF | ▃ | U+2583 |
| CR | ▄ | U+2584 |
| エスケープ | ▀ | U+2580 |

メタ文字自体が入力に含まれる場合は `▀` でエスケープされます（例: `▁` → `▀▁`）。

---

### CRF 重み管理

従来は CRF の遷移重みがハードコードされていましたが、現在は **コード変更なし**に調整できます。

#### A. 設定ファイルで上書き

```bash
./tools/mmjp_train \
  --corpus your_corpus.txt \
  --out model.bin \
  --crf_config crf_weights.cfg
```

`crf_weights.cfg` 例：

```ini
# transitions
trans00 = 0.2
trans01 = -0.4
trans10 = 0.0
trans11 = -0.6

# feature weights: feat <tid> <label> <v1> <v2> = <weight>
feat 1 1 250 0 = 2.0   # prev=BOS, label=1
```

#### B. 教師あり最適化（L-BFGS / SGD）

```bash
./tools/mmjp_train \
  --corpus your_corpus.txt \
  --out model.bin \
  --crf_supervised gold_segmented.txt \
  --crf_opt lbfgs \
  --crf_epochs 30 \
  --crf_l2 1e-4
```

`gold_segmented.txt` の形式（空白区切り）：

```text
東京 都 に 住んで い ます 。
明日 は 雨 かも しれ ない 。
```

#### C. 教師なし最適化

LM Viterbi の結果を疑似ラベルとして CRF 重みを最適化：

```bash
./tools/mmjp_train \
  --corpus your_corpus.txt \
  --out model.bin \
  --crf_unsupervised 1 \
  --crf_unsup_sentences 1000
```

---

### Subword Regularization（確率的分割）

内部でラティスを持っているため、確率的に分割を揺らがせることで LLM 学習データ生成時の **Subword Regularization** に使えます。

#### FFBS サンプリング

```bash
echo '東京都に住んでいます。' | ./tools/mmjp_tokenize \
  --model model.bin \
  --sample \
  --temperature 1.2 \
  --nsamples 5
```

#### N-best からサンプル

```bash
echo '東京都に住んでいます。' | ./tools/mmjp_tokenize \
  --model model.bin \
  --sample_nbest 8 \
  --nsamples 5
```

#### LLM 用データ生成レシピ

```bash
# ベスト（決定的）
cat input.txt | ./tools/mmjp_tokenize --model model.bin > out.best.txt

# サンプル（揺らぎ）
cat input.txt | ./tools/mmjp_tokenize --model model.bin --sample --temperature 1.2 --nsamples 2 > out.sample.txt

# 両方を混ぜて学習データにする
```

---

### 言語非依存の文字種分類

従来の日本語ハードコード（ひらがな・カタカナ・漢字）を廃止し、設定可能な文字種分類モードを追加しました。

| モード | 説明 |
|--------|------|
| `compat` | 従来互換（日本語ハードコード）- デフォルト |
| `ascii` | ASCII 範囲のみ分類、非ASCIIはOTHER |
| `utf8len` | UTF-8 バイト長で分類（1/2/3/4 バイト文字） |
| `ranges` | ユーザー定義の Unicode 範囲で分類 |

```bash
./tools/mmjp_train \
  --corpus your_corpus.txt \
  --out model.bin \
  --cc_mode ranges \
  --cc_ranges ranges.txt \
  --cc_fallback utf8len
```

`ranges.txt` 例：

```text
0x3040 0x309F 4   # ひらがな
0x30A0 0x30FF 5   # カタカナ
0x4E00 0x9FFF 6   # 漢字（CJK統合漢字）
```

---

## CLI リファレンス

### mmjp_train（学習）

```bash
./tools/mmjp_train --corpus corpus.txt --out model.bin [options]
```

| オプション | デフォルト | 説明 |
|------------|------------|------|
| `--corpus PATH` | (必須) | 学習コーパス（1行1文、UTF-8） |
| `--out PATH` | (必須) | 出力モデルファイル |
| `--vocab N` | 8000 | 目標語彙サイズ |
| `--max_piece_len N` | 8 | 最大ピース長 |
| `--iters N` | 5 | EM イテレーション回数 |
| `--lossless_ws 0\|1` | 0 | 可逆空白エンコード |
| `--lossless_eol 0\|1` | 0 | 行末メタ LF 付与 |
| `--crf_supervised PATH` | - | 教師データ |
| `--crf_unsupervised 0\|1` | 0 | 教師なし学習 |
| `--crf_opt sgd\|lbfgs` | lbfgs | 最適化手法 |
| `--crf_epochs N` | 20 | エポック数 |
| `--cc_mode MODE` | compat | 文字種モード |

### mmjp_tokenize（推論）

```bash
./tools/mmjp_tokenize --model model.bin [options]
```

| オプション | デフォルト | 説明 |
|------------|------------|------|
| `--model PATH` | (必須) | モデルファイル |
| `--lossless_ws N` | -1 | -1=自動、0=オフ、1=オン |
| `--read_all 1` | 0 | stdin 全体を1テキストとして処理 |
| `--detok` | - | デトークナイズモード |
| `--sample` | - | FFBS サンプリング |
| `--temperature X` | 1.0 | サンプリング温度 |
| `--nsamples N` | 1 | 出力サンプル数 |
| `--nbest N` | - | N-best 出力 |
| `--sample_nbest N` | - | top-N からサンプル |

### mmjp_export_c（MCU 用エクスポート）

model.bin を C ヘッダファイルに変換（組み込み用）。

```bash
./tools/mmjp_export_c --model model.bin --out model.h --symbol mmjp
```

---

## Python バインディング

```bash
pip install .
```

```python
import mmjp

m = mmjp.Model("models/mmjp_wiki.bin")

# 基本
print(m.tokenize("東京都に住んでいます。"))

# サンプリング
print(m.sample("東京都に住んでいます。", temperature=1.0, seed=1))

# N-best
for cand in m.nbest("東京都に住んでいます。", nbest=8):
    print(cand)

# オフセット付き
print(m.tokenize_with_offsets("東京都に住んでいます。", unit="char"))
```

---

## 実装メモ

| コンポーネント | パス |
|----------------|------|
| 推論コア | `npycrf_lite/` |
| モデル I/O | `tools/mmjp_model.c/h` |
| 学習ツール | `tools/mmjp_train.c` |
| トークナイザ CLI | `tools/mmjp_tokenize.c` |
| MCU エクスポート | `tools/mmjp_export_c.c` |
| Lossless エンコード | `mmjp_lossless.c/h` |
| Python 拡張 | `mmjp/_mmjp.c` |

### アルゴリズム

- 線形連鎖 CRF（Viterbi / Forward-Backward / FFBS / N-best）
- Unigram Language Model によるサブワード分割（SentencePiece の Unigram と同系統）
- Double-array trie

---

## ライセンス

- **コード**: MIT License（`LICENSE`）
- **モデル**: 学習コーパスのライセンスに依存
  - 同梱 `models/mmjp_wiki.bin` は Wikipedia（CC BY-SA）で学習
