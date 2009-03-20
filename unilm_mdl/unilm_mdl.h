/*
 * unilm_mdl.h
 *
 * ユニグラム言語モデル学習/推論コア (C99)
 *
 * この ライブラリは「マイコン向け」制約を想定:
 *   - UTF-8 はバイト列として扱うが、DP 位置は UTF-8 コードポイント境界
 *   - *_init_static で全バッファを提供すればヒープアロケーション不要
 *   - トークンマッチングには付属の double_array_trie.* (ダブル配列トライ) を使用
 *     ピース ID は終端ノードの base[...] に負の値として格納
 *
 * スコープ
 *   - ユニグラムトークナイゼーション確率の EM (前向き-後ろ向き) 最適化
 *   - シンプルな MDL スタイルの枝刈り:
 *       ピースが文字フォールバックに比べて記述長を削減するなら保持
 *       （または、そのスコアで上位 K を保持）
 *
 * 対象外
 *   - 生コーパスからの大規模候補生成（SentencePiece スタイル）
 *     候補リストは別途用意し（接尾辞配列/ヒューリスティクスなどで）、学習器を呼び出す
 */

#ifndef UNILM_MDL_H
#define UNILM_MDL_H

#include <stddef.h>
#include <stdint.h>

#include "double_array_trie.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ---------------- 数値型 ---------------- */

#ifndef UNILM_REAL_T
#define UNILM_REAL_T double
#endif

typedef UNILM_REAL_T unilm_real_t;

/* ---------------- エラーコード ---------------- */

enum {
  UNILM_OK = 0,
  UNILM_ERR_BADARG   = -1,   /* 不正な引数 */
  UNILM_ERR_NOMEM    = -2,   /* メモリ確保失敗 */
  UNILM_ERR_FULL     = -3,   /* 容量不足 */
  UNILM_ERR_UTF8     = -4,   /* UTF-8 エラー */
  UNILM_ERR_NOCOVER  = -5,   /* 与えられた語彙で文がトークナイズできない */
  UNILM_ERR_RANGE    = -6,   /* バッファ/ワークスペースが小さすぎる */
  UNILM_ERR_INTERNAL = -7,   /* 内部エラー */
  UNILM_ERR_IO       = -8    /* I/O エラー */
};

/* ---------------- コーパスイテレータ ---------------- */

/*
 * next():
 *   文が利用可能な場合は 1 を返し、(*out, *out_len) を設定
 *   終端で 0 を返す
 *   エラー時は < 0 を返す
 * reset(): (オプション) 次の EM 反復のために先頭に巻き戻す
 */

typedef struct {
  int (*next)(void *user, const uint8_t **out, size_t *out_len);
  void (*reset)(void *user);
  void *user;
} unilm_corpus_iter_t;

/* ヘルパー: (ptr, len) の配列に対するイテレータ */

typedef struct {
  const uint8_t *const *sent;  /* 文ポインタの配列 */
  const size_t *sent_len;      /* 文長の配列 */
  size_t n;                    /* 文の総数 */
  size_t i;                    /* 現在位置 */
} unilm_array_corpus_t;

int unilm_array_corpus_next(void *user, const uint8_t **out, size_t *out_len);
void unilm_array_corpus_reset(void *user);

/* ---------------- モデル ---------------- */

enum {
  UNILM_PIECE_MANDATORY = 1u << 0  /* 枝刈りされない（必須ピース） */
};

/* ピース情報 */
typedef struct {
  uint32_t str_off;  /* model->strbuf 内のオフセット */
  uint16_t len;      /* バイト長 */
  uint16_t len_cp;   /* UTF-8 コードポイント数 */
  uint8_t  flags;    /* フラグ (UNILM_PIECE_MANDATORY など) */
  uint8_t  reserved;
} unilm_piece_t;

/* ユニグラムモデル構造体 */
typedef struct {
  /* ピース文字列ストレージ（バイトプール） */
  uint8_t *strbuf;
  size_t strbuf_cap;
  size_t strbuf_len;

  /* 語彙 */
  unilm_piece_t *pieces;     /* ピース情報配列 */
  unilm_real_t *logp;        /* ln(確率) */
  size_t vocab_size;         /* 現在の語彙サイズ */
  size_t vocab_cap;          /* 語彙容量 */

  /* マッチャー（ダブル配列トライ）。終端ノードの base には -(id+1) を格納 */
  da_trie_t trie;

  /* 所有権フラグ */
  int dynamic;
} unilm_model_t;

/* 動的アロケーションで初期化 */
int unilm_model_init_dynamic(unilm_model_t *m,
                            size_t vocab_cap,
                            size_t strbuf_cap,
                            size_t da_cap);

/* 静的バッファで初期化 */
int unilm_model_init_static(unilm_model_t *m,
                           uint8_t *strbuf, size_t strbuf_cap,
                           unilm_piece_t *pieces, unilm_real_t *logp, size_t vocab_cap,
                           da_index_t *da_base, da_index_t *da_check, size_t da_cap);

/* 内部バッファを解放（動的モードのみ） */
void unilm_model_free(unilm_model_t *m);

/* モデルをクリア（バッファは保持） */
int unilm_model_clear(unilm_model_t *m);

/* ピースを追加。ID (>=0) またはエラー (<0) を返す */
int32_t unilm_model_add_piece(unilm_model_t *m,
                             const uint8_t *bytes, size_t len,
                             uint8_t flags);

/* 既存 ID を検索。ID (>=0) または見つからなければ -1 を返す */
int32_t unilm_model_find_id(const unilm_model_t *m,
                            const uint8_t *bytes, size_t len);

/* ピースのバイト列ポインタを取得 */
const uint8_t *unilm_model_piece_bytes(const unilm_model_t *m, size_t id, size_t *len_out);

/* 便利関数: logp を設定して正規化 */
int unilm_model_set_logp(unilm_model_t *m, uint32_t id, unilm_real_t logp);
int unilm_model_normalize(unilm_model_t *m, unilm_real_t min_prob);

/*
 * 内部のダブル配列トライを再構築する。
 *
 * 語彙が追加・削除・並び替えされた後にトライを作り直したい場合に利用する。
 * ピース文字列（バイト列）の辞書順に挿入して構築することで、比較的コンパクトなトライになりやすい。
 */
int unilm_model_rebuild_trie_sorted(unilm_model_t *m);

/* ---------------- ワークスペース ---------------- */

typedef struct {
  /* UTF-8 コードポイント境界オフセット。容量: max_codepoints+1 */
  uint32_t *cp_off;
  size_t cp_off_cap;

  /* DP 配列。容量: max_codepoints+1 */
  unilm_real_t *alpha;   /* 前向き確率 */
  unilm_real_t *beta;    /* 後ろ向き確率 */
  size_t dp_cap;

  /* ビタビバックトレース配列。容量: max_codepoints+1 */
  int32_t *bp_prev;      /* 前のコードポイント位置 */
  int32_t *bp_piece;     /* 選択されたピース ID */
  size_t bp_cap;

  /* 枝刈りサポート */
  uint8_t *keep;         /* 保持フラグ配列 */
  size_t keep_cap;
  uint32_t *heap_idx;    /* ヒープインデックス配列 */
  unilm_real_t *heap_score; /* ヒープスコア配列 */
  size_t heap_cap;

  int dynamic;
} unilm_workspace_t;

/* 動的アロケーションでワークスペースを初期化 */
int unilm_workspace_init_dynamic(unilm_workspace_t *wk,
                                size_t max_codepoints,
                                size_t vocab_cap,
                                size_t heap_cap);

/* 静的バッファでワークスペースを初期化 */
int unilm_workspace_init_static(unilm_workspace_t *wk,
                               uint32_t *cp_off, size_t cp_off_cap,
                               unilm_real_t *alpha, unilm_real_t *beta, size_t dp_cap,
                               int32_t *bp_prev, int32_t *bp_piece, size_t bp_cap,
                               uint8_t *keep, size_t keep_cap,
                               uint32_t *heap_idx, unilm_real_t *heap_score, size_t heap_cap);

/* ワークスペースを解放（動的モードのみ） */
void unilm_workspace_free(unilm_workspace_t *wk);

/* ---------------- 学習 ---------------- */

/* 学習設定 */
typedef struct {
  int num_iters;             /* EM 反復回数 */
  int max_piece_len_cp;      /* DP マッチ探索の上限（コードポイント数） */
  unilm_real_t smoothing;    /* 擬似カウント（例: 0.1） */

  /* MDL パラメータ (nat 単位): model_cost = lambda0 + lambda_len * len_cp */
  unilm_real_t mdl_lambda0;
  unilm_real_t mdl_lambda_len;

  /* > 0 の場合、このサイズに枝刈り（必須ピースは常に保持） */
  size_t target_vocab_size;

  /* 非ゼロの場合、各 EM 反復後に枝刈り */
  int prune_each_iter;

  /* log(0) を避けるための数値フロア（例: 1e-12） */
  unilm_real_t min_prob;
} unilm_train_config_t;

/* EM 統計 */
typedef struct {
  unilm_real_t loglik;        /* sum log P(文) */
  unilm_real_t n_sent;        /* 文の数 */
  unilm_real_t n_tokens_exp;  /* 期待トークン数 */
} unilm_em_stats_t;

/* E ステップ: 期待カウント + 対数尤度。counts サイズ >= model->vocab_size（内部でクリア） */
int unilm_em_e_step(const unilm_model_t *m,
                    const unilm_corpus_iter_t *it,
                    const unilm_train_config_t *cfg,
                    unilm_workspace_t *wk,
                    unilm_real_t *counts,
                    unilm_em_stats_t *out_stats);

/* M ステップ: カウントから logp を更新 */
int unilm_em_m_step(unilm_model_t *m,
                    const unilm_train_config_t *cfg,
                    const unilm_real_t *counts);

/* MDL 枝刈り。UNILM_OK または < 0 を返す */
int unilm_prune_mdl(unilm_model_t *m,
                    const unilm_train_config_t *cfg,
                    unilm_workspace_t *wk,
                    const unilm_real_t *counts);

/* 完全ループ (E+M+オプション枝刈り) */
int unilm_train_em_mdl(unilm_model_t *m,
                       const unilm_corpus_iter_t *it,
                       const unilm_train_config_t *cfg,
                       unilm_workspace_t *wk,
                       unilm_real_t *counts,
                       unilm_em_stats_t *out_last_stats);

/* ---------------- 推論 ---------------- */

/* ビタビトークナイゼーション（最大確率）。out_ids にピース ID を出力 */
int unilm_viterbi_tokenize(const unilm_model_t *m,
                           const uint8_t *sent, size_t sent_len,
                           int max_piece_len_cp,
                           unilm_workspace_t *wk,
                           uint32_t *out_ids, size_t out_cap,
                           size_t *out_n);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* UNILM_MDL_H */
