# MMJP パラメータ探索（λ0 × mdl_lambda_len）Pareto表

## 条件
- unlabeled: Wikipedia文 20k（`unlabeled_train_20k.txt`）
- labeled: Janome銀ラベル train 800文（`labeled_train_800.txt`）
- 評価: Janome銀ラベル test 181文（`raw_test_181.txt` と `labeled_test_181_nospacegold.txt`）
- 指標: 境界F1（単語開始位置）
- 速度: 20k文を `mmjp_tokenize` で処理した wall time（/usr/bin/time -v）
- vocab=8000, max_piece_len=8, iters=5, CRF: lbfgs epochs=30 l2=1e-4

## 重要な観察
- **今回の条件では `mdl_lambda_len` を 0.15〜0.4 に動かしても、境界F1/分割粒度は変化しませんでした**（同一λ0なら同じ分割になりました）。
  - 実質的にチューニングで効いているのは **`lambda0`（CRFとLMの混合比）** でした。
  - `mdl_lambda_len` が効くかどうかは、候補piece数・vocab上限・コーパスによって変わる可能性があります。

## Pareto（F1 ↔ 分割粒度 ↔ 速度）: λ0 ごとの集約

| lambda0 | F1 | P | R | avg tokens/sent | avg token len | tok 20k time (s) | ± | throughput (k sent/s) | tok RSS (MB) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.0 | 0.6399 | 0.9570 | 0.4806 | 15.15 | 3.52 | 0.645 | 0.045 | 31.0 | 4.60 |
| 0.7 | 0.7099 | 0.9621 | 0.5624 | 17.64 | 3.02 | 0.660 | 0.042 | 30.3 | 4.05 |
| 0.5 | 0.7781 | 0.9615 | 0.6534 | 20.51 | 2.60 | 0.625 | 0.039 | 32.0 | 4.64 |
| 0.3 | 0.8245 | 0.9303 | 0.7404 | 24.02 | 2.22 | 0.620 | 0.018 | 32.2 | 4.31 |
| 0.2 | 0.8359 | 0.8913 | 0.7869 | 26.64 | 2.00 | 0.633 | 0.013 | 31.6 | 3.97 |

## 推奨レシピ（今回の条件で）
- **最大F1**: `lambda0=0.2`（F1=0.8359 / avg tokens/sent=26.64）
- **バランス**: `lambda0=0.3`（F1=0.8245 / avg tokens/sent=24.02）

### 学習コマンド（mdl_lambda_len は今回効かなかったので 0.15 のままでOK）
```bash
./tools/mmjp_train \
  --corpus unlabeled_train_20k.txt \
  --out mmjp_crf.bin \
  --vocab 8000 --max_piece_len 8 --iters 5 \
  --mdl_lambda_len 0.15 \
  --crf_supervised labeled_train_800.txt \
  --crf_opt lbfgs --crf_epochs 30 --crf_l2 1e-4
```

### 推論時（λ0だけ変えたい場合）
- 今回のMMJP実装では `lambda0` は **モデル.bin内の値**として使われ、tokenize時にオプション上書きはできません。
- そのため λ0 をスイープするには、
  1) `--lambda0` を変えて学習し直す、または
  2) **モデル.binのヘッダ中のλ0（Q8.8 int16, オフセット32）をパッチする**
  のどちらかになります（今回の探索では後者で高速化しました）。

## 付録
- フルグリッド（20行）: `mmjp_lambda_len_grid.tsv`
- λ0ごとの集約CSV: `mmjp_pareto_collapsed.csv`
