/*
 * npycrf_lite.h
 *
 * UTF-8分かち書き用コンパクトC99ライブラリ
 * NPYCRFアプローチ（マルコフCRF＋生成モデルを半マルコフラティス上で統合）に
 * インスパイアされた実装。
 *
 * 設計方針:
 *  - 推論特化実装（ビタビ/max-sum デコード）
 *  - 学習（JESS-CM/NPYLMのギブスサンプリング）はMCU向けスコープ外
 *    →オフライン学習でコンパクトなテーブル（CRF素性重み＋LM確率）を出荷
 *  - CRF部は2状態マルコフモデル（"単語開始"=1, "単語内部"=0）
 *    →単語単位のセグメントスコアに変換
 *  - 生成モデルはコンパクトな辞書LM（ユニグラム＋オプションでバイグラムバックオフ）
 *
 * 依存:
 *  - オプション: double_array_trie.h/.c（同梱）を辞書格納に使用
 *
 * メモリ:
 *  - 動的メモリ確保不要（ユーザー提供バッファを使用）
 *  - 固定小数点演算でFPU不要
 */

#pragma once

#include <stddef.h>
#include <stdint.h>

#include "double_array_trie.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ======================================================================
 * 固定小数点設定
 * ====================================================================== */

/*
 * 全スコアは符号付き固定小数点整数で表現。
 * デフォルト: Q8.8（スケール=256）
 *
 * Q8.8形式:
 *  - 上位8ビット: 整数部
 *  - 下位8ビット: 小数部
 *  - 例: 1.5 = 0x0180 (256 + 128)
 */
#ifndef NPYCRF_Q
#define NPYCRF_Q 8
#endif

typedef int32_t npycrf_score_t;

/* スケーリング係数（1.0に相当する整数値） */
#define NPYCRF_SCORE_SCALE   (1 << NPYCRF_Q)

/* 負の無限大を表す番兵値（DP初期化用） */
#define NPYCRF_SCORE_NEG_INF ((npycrf_score_t)(-0x3fffffff))

/* 浮動小数点リテラルを固定小数点に変換（サンプルコード用の便利マクロ） */
#define NPYCRF_F2Q(x) ((npycrf_score_t)((x) * (double)NPYCRF_SCORE_SCALE))

/* ======================================================================
 * 文字クラスID（推論側 char_class() と一致させる）
 * ====================================================================== */

/*
 * NOTE:
 *  - これらのIDは CRF 素性キー (npycrf_feat_key) に直接埋め込まれるため、
 *    ツール側（学習/重み調整）と推論側で必ず一致している必要があります。
 *  - 互換性のため、BOS/EOS は 250/251 を予約します。
 */

#define NPYCRF_CC_OTHER     0u
#define NPYCRF_CC_SPACE     1u
#define NPYCRF_CC_DIGIT     2u
#define NPYCRF_CC_ALPHA     3u
#define NPYCRF_CC_HIRAGANA  4u
#define NPYCRF_CC_KATAKANA  5u
#define NPYCRF_CC_KANJI     6u
#define NPYCRF_CC_FULLWIDTH 7u
#define NPYCRF_CC_SYMBOL    8u
#define NPYCRF_CC_BOS       250u
#define NPYCRF_CC_EOS       251u

/* UTF-8バイト長ベースの文字クラス（UTF8LENモード用） */
#define NPYCRF_CC_UTF8_2BYTE  9u   /* 2バイト文字 (U+0080-U+07FF) */
#define NPYCRF_CC_UTF8_3BYTE  10u  /* 3バイト文字 (U+0800-U+FFFF) */
#define NPYCRF_CC_UTF8_4BYTE  11u  /* 4バイト文字 (U+10000-U+10FFFF) */

/* ======================================================================
 * 文字種分類設定（言語非依存化）
 * ====================================================================== */

/*
 * 文字種分類モード
 *
 * NPYCRF_CC_MODE_ASCII:   ASCIIのみ分類、非ASCIIはOTHER
 * NPYCRF_CC_MODE_UTF8LEN: UTF-8バイト長でバケット分類
 * NPYCRF_CC_MODE_RANGES:  ユーザ定義のUnicode範囲表
 * NPYCRF_CC_MODE_COMPAT:  後方互換（日本語ハードコード）
 */
typedef enum {
  NPYCRF_CC_MODE_ASCII   = 0,
  NPYCRF_CC_MODE_UTF8LEN = 1,
  NPYCRF_CC_MODE_RANGES  = 2,
  NPYCRF_CC_MODE_COMPAT  = 3
} npycrf_cc_mode_t;

/*
 * Unicode範囲→文字クラスIDのマッピング
 *
 * [lo, hi] の範囲にあるコードポイントに class_id を割り当てる
 */
typedef struct {
  uint32_t lo;       /* 範囲開始 (inclusive) */
  uint32_t hi;       /* 範囲終了 (inclusive) */
  uint8_t  class_id; /* 文字クラスID */
  uint8_t  _pad[3];  /* アラインメント用パディング */
} npycrf_cc_range_t;

/*
 * 文字種分類設定
 *
 * mode:        分類モード
 * fallback:    RANGESモードで一致しない場合のフォールバック
 * ranges:      Unicode範囲配列（昇順ソート推奨）
 * range_count: ranges配列の要素数
 */
typedef struct {
  npycrf_cc_mode_t mode;
  npycrf_cc_mode_t fallback;
  const npycrf_cc_range_t *ranges;
  uint32_t range_count;
} npycrf_cc_t;

/*
 * コードポイントから文字クラスを判定（言語非依存版）
 *
 * ASCII範囲(0-127)は従来通り分類し、非ASCIIはmodeに応じて分類する。
 *
 * @param cc  文字種設定（NULLの場合はUTF8LENモードとして動作）
 * @param cp  Unicodeコードポイント
 * @return 文字クラスID
 */
uint8_t npycrf_char_class_cp(const npycrf_cc_t *cc, uint32_t cp);

/* ======================================================================
 * ID定義
 * ====================================================================== */

/*
 * 単語IDは16ビット（テーブルサイズ削減のため）
 * 最大65533語彙（0xFFFE, 0xFFFFは予約）
 */
typedef uint16_t npycrf_id_t;

/* 未知語/OOV を示す特殊ID */
#define NPYCRF_ID_NONE ((npycrf_id_t)0xFFFFu)

/* 文頭(BOS: Beginning Of Sentence)を示す特殊ID */
#define NPYCRF_ID_BOS  ((npycrf_id_t)0xFFFEu)

/* ======================================================================
 * CRFモデル構造体
 * ====================================================================== */

/*
 * 2状態マルコフCRF（"単語開始"ラベリング用）
 *
 * ラベル定義:
 *  label=1: 単語開始位置
 *  label=0: 単語内部（非開始位置）
 *
 * 遷移重み:
 *  - 観測に依存しない定数（コンパクト性のため）
 *  - 例: trans01 = label 0 から label 1 への遷移重み
 *
 * 放射重み:
 *  - ソート済みキーテーブルから二分探索で検索
 */
typedef struct {
  /* 遷移重み（Q8.8形式）: w(前ラベル → 次ラベル) */
  int16_t trans00;  /* 内部 → 内部 */
  int16_t trans01;  /* 内部 → 開始 */
  int16_t trans10;  /* 開始 → 内部 */
  int16_t trans11;  /* 開始 → 開始 */

  /* BOS から最初のラベル(=1)への遷移重み（Q8.8） */
  int16_t bos_to1;

  /* 放射素性テーブル（キーでソート済み） */
  const uint32_t *feat_key;  /* 昇順ソート済みキー配列 */
  const int16_t  *feat_w;    /* 対応する重み（Q8.8） */
  uint32_t feat_count;       /* テーブルエントリ数 */
} npycrf_crf_t;

/*
 * 放射素性キーのパック形式
 *
 * 素性は (template_id, label, v1, v2) の4要素で定義。
 * これを32ビットキーにパック:
 *   key = (template_id << 24) | (label << 16) | (v1 << 8) | v2
 *
 * デフォルトテンプレート（本ライブラリで使用）:
 *   0: CUR_CLASS(v1)        - 現在位置の文字クラス
 *   1: PREV_CLASS(v1)       - 前位置の文字クラス
 *   2: NEXT_CLASS(v1)       - 次位置の文字クラス
 *   3: PREV_CUR_CLASS(v1, v2) - 前・現在の文字クラスペア
 *   4: CUR_NEXT_CLASS(v1, v2) - 現在・次の文字クラスペア
 *
 * 文字クラス例: ひらがな、カタカナ、漢字、アルファベット、数字、空白など
 */
static inline uint32_t npycrf_feat_key(uint8_t template_id, uint8_t label, uint8_t v1, uint8_t v2) {
  return ((uint32_t)template_id << 24) | ((uint32_t)label << 16) | ((uint32_t)v1 << 8) | (uint32_t)v2;
}

/* ======================================================================
 * 言語モデル（LM）構造体
 * ====================================================================== */

/*
 * 辞書言語モデル
 *
 * コンポーネント:
 *  - trie: UTF-8単語バイト列 → 単語ID のマッピング
 *    （終端ノードのBASE値を負数にエンコード）
 *  - unigram: ID毎の対数確率テーブル（Q8.8）
 *  - bigram: (前ID, 現ID)ペアの対数確率テーブル（オプション）
 *  - unknown: 未知語ペナルティ = unk_base + unk_per_cp * コードポイント数
 *
 * バイグラムテーブル:
 *  - キーでソート済み: key = (prev_id << 16) | curr_id
 *  - 二分探索でルックアップ
 *  - 見つからない場合はユニグラムにバックオフ
 */
typedef struct {
  da_trie_ro_t trie;  /* 読み取り専用ダブル配列トライ */

  const int16_t *logp_uni;   /* ユニグラム対数確率（Q8.8）, サイズ=vocab_size */
  uint32_t vocab_size;       /* 語彙サイズ */

  const uint32_t *bigram_key;  /* バイグラムキー（ソート済み） */
  const int16_t  *logp_bi;     /* バイグラム対数確率（Q8.8） */
  uint32_t bigram_size;        /* バイグラムエントリ数 */

  int16_t unk_base;    /* 未知語基本ペナルティ（Q8.8） */
  int16_t unk_per_cp;  /* 未知語・コードポイント毎ペナルティ（Q8.8, 通常負値） */
} npycrf_lm_t;

/* ======================================================================
 * 統合モデル構造体
 * ====================================================================== */

/*
 * モデルフラグ
 */
#define NPYCRF_FLAG_LOSSLESS_WS  (1u << 0)  /* lossless空白変換が有効 */
#define NPYCRF_FLAG_CC_ASCII     (1u << 8)  /* cc_mode = ASCII */
#define NPYCRF_FLAG_CC_UTF8LEN   (1u << 9)  /* cc_mode = UTF8LEN */
#define NPYCRF_FLAG_CC_RANGES    (1u << 10) /* cc_mode = RANGES */
#define NPYCRF_FLAG_CC_COMPAT    (1u << 11) /* cc_mode = COMPAT */

/*
 * CRF + LM 統合モデル
 *
 * 統合スコア計算:
 *   score = crf_score + lambda0 * lm_score
 *
 * パラメータ:
 *  - lambda0: 生成モデル（LM）の重み係数（Q8.8）
 *    大きいほどLMを重視、小さいほどCRFを重視
 *  - max_word_len: 最大単語長（UTF-8コードポイント単位）
 *  - flags: モデルフラグ（lossless等）
 *  - cc: 文字種分類設定
 */
typedef struct {
  npycrf_crf_t crf;  /* CRFモデル */
  npycrf_lm_t  lm;   /* 言語モデル */

  int16_t lambda0;       /* LM重み係数（Q8.8） */
  uint16_t max_word_len; /* 最大単語長（コードポイント数） */
  uint32_t flags;        /* モデルフラグ */
  npycrf_cc_t cc;        /* 文字種分類設定 */
} npycrf_model_t;

/* ======================================================================
 * ワークスペース構造体（動的メモリ確保不要）
 * ====================================================================== */

/*
 * デコード用作業領域
 *
 * 設計:
 *  - ユーザー提供バッファを内部で分割して使用
 *  - malloc/free不要でMCU環境に適合
 *
 * 内部テーブル:
 *  - cp_off: コードポイント→バイトオフセットマッピング
 *  - emit0/emit1: ラベル0/1の放射スコア（事前計算）
 *  - pref_emit0: emit0の累積和（区間和の高速計算用）
 *  - span_id/span_luni: スパン(終了位置,長さ)毎の単語ID・ユニグラムスコア
 *  - bp_prevlen: バックポインタ（ビタビ経路復元用）
 *  - dp_ring: DPリングバッファ（メモリ効率化）
 */
typedef struct {
  uint16_t max_n_cp;      /* 最大コードポイント数 */
  uint16_t max_word_len;  /* 最大単語長 */

  /* ユーザー提供バッファへのポインタ群 */
  uint16_t *cp_off;      /* [max_n_cp+1] バイトオフセット配列 */
  int16_t  *emit0;       /* [max_n_cp] ラベル0放射スコア（Q8.8） */
  int16_t  *emit1;       /* [max_n_cp] ラベル1放射スコア（Q8.8） */
  int32_t  *pref_emit0;  /* [max_n_cp+1] emit0累積和（Q8.8） */

  /* スパンテーブル: (終了位置, 長さ) でインデックス、行優先 */
  npycrf_id_t *span_id;   /* [(max_n_cp+1)*(max_word_len+1)] 単語ID */
  int16_t     *span_luni; /* [(max_n_cp+1)*(max_word_len+1)] ユニグラム/OOVスコア */

  /* バックポインタ: 前単語の長さ (0..L) */
  uint8_t *bp_prevlen;   /* [(max_n_cp+1)*(max_word_len+1)] */

  /* DPリングバッファ: 位置 mod (L+1)、最終長さ 0..L */
  npycrf_score_t *dp_ring; /* [(max_word_len+1)*(max_word_len+1)] */
} npycrf_work_t;

/*
 * 必要なワークバッファサイズを計算
 *
 * @param max_n_cp      最大コードポイント数
 * @param max_word_len  最大単語長
 * @return 必要バイト数
 */
size_t npycrf_workbuf_size(uint16_t max_n_cp, uint16_t max_word_len);

/*
 * ワーク構造体を初期化（バッファを内部で分割）
 *
 * @param w             ワーク構造体へのポインタ
 * @param buf           ユーザー提供バッファ
 * @param buf_size      バッファサイズ
 * @param max_n_cp      最大コードポイント数
 * @param max_word_len  最大単語長
 * @return 0=成功, 負数=エラー
 */
int npycrf_work_init(npycrf_work_t *w, void *buf, size_t buf_size,
                     uint16_t max_n_cp, uint16_t max_word_len);

/* ======================================================================
 * UTF-8ヘルパー関数
 * ====================================================================== */

/*
 * UTF-8バッファからコードポイント開始オフセット配列を構築
 *
 * @param utf8     UTF-8バイト列
 * @param len      バイト長
 * @param out_off  出力オフセット配列（要 n_cp+1 以上の容量）
 * @param out_cap  出力配列の容量
 * @return コードポイント数（n_cp）、エラー時は0
 *
 * 出力形式:
 *  - out_off[i] = i番目コードポイントの開始バイト位置
 *  - out_off[n_cp] = len（終端）
 */
size_t npycrf_utf8_make_offsets(const uint8_t *utf8, size_t len,
                                uint16_t *out_off, size_t out_cap);

/*
 * 境界インデックスをコードポイント単位からバイト単位に変換
 *
 * @param cp_off      コードポイントオフセット配列
 * @param b_cp        境界配列（コードポイント単位）
 * @param b_count     境界数
 * @param b_bytes_out 出力境界配列（バイト単位）
 */
void npycrf_boundaries_cp_to_bytes(const uint16_t *cp_off,
                                  const uint16_t *b_cp, size_t b_count,
                                  uint16_t *b_bytes_out);

/* ======================================================================
 * ダブル配列トライ値ヘルパー
 * ====================================================================== */

/*
 * 終端値（単語ID）を設定
 *
 * 終端ノードのBASE値を負数としてエンコード: base[term] = -(id + 1)
 * 書き込み可能/動的トライ (da_trie_t) が必要。
 *
 * @param da       書き込み可能トライ
 * @param key      キー（UTF-8バイト列）
 * @param key_len  キー長
 * @param id       設定する単語ID
 * @return DA_OK=成功, その他=エラー
 */
int npycrf_da_set_term_value(da_trie_t *da, const uint8_t *key, size_t key_len, npycrf_id_t id);

/*
 * 終端値（単語ID）を取得
 *
 * @param da       読み取り専用トライ
 * @param key      キー（UTF-8バイト列）
 * @param key_len  キー長
 * @param out_id   出力単語ID
 * @return 1=発見, 0=未発見
 */
int npycrf_da_ro_get_term_value(const da_trie_ro_t *da, const uint8_t *key, size_t key_len, npycrf_id_t *out_id);

/* ======================================================================
 * デコードAPI
 * ====================================================================== */

/*
 * ビタビアルゴリズムによる分かち書き境界デコード
 *
 * 半マルコフラティス上でCRF+LM統合スコアを最大化する
 * 分割を探索。
 *
 * @param model          統合モデル
 * @param utf8           入力UTF-8文字列
 * @param len            入力バイト長
 * @param work           ワークスペース
 * @param out_b_cp       出力境界配列（コードポイントインデックス）
 * @param out_b_cap      出力配列容量
 * @param out_b_count    出力境界数（= トークン数 + 1）
 * @param out_best_score 最良スコア出力（Q8.8, NULLで省略可）
 * @return 0=成功, 負数=エラー
 *
 * 出力境界形式:
 *  - [0, b1, b2, ..., n_cp] のように0から始まりn_cpで終わる
 *  - トークン i は境界 [out_b_cp[i], out_b_cp[i+1]) の範囲
 *
 * 使用例:
 *   "東京都" → 境界 [0, 3] → 1トークン "東京都"
 *   "東京|都" → 境界 [0, 2, 3] → 2トークン "東京", "都"
 */
int npycrf_decode(const npycrf_model_t *model,
                  const uint8_t *utf8, size_t len,
                  npycrf_work_t *work,
                  uint16_t *out_b_cp, size_t out_b_cap,
                  size_t *out_b_count,
                  npycrf_score_t *out_best_score);

/* ======================================================================
 * Subword Regularization（確率的分割）
 * ====================================================================== */

/*
 * 追加ワークバッファサイズ（Forward-Filtering Backward-Sampling 用）
 *
 * サンプリングは推論の「最良解」ではなく、スコアに比例した確率分布から
 * 分割を1つサンプルします（温度パラメータ対応）。
 */
size_t npycrf_samplebuf_size(uint16_t max_n_cp, uint16_t max_word_len);

/*
 * Forward-Filtering Backward-Sampling による1サンプルデコード
 *
 * @param temperature  1.0=通常、>1 で分割が揺らぎやすくなる、<1 で鋭くなる
 * @param seed         乱数シード（xorshift32）
 *
 * NOTE: 確率計算は double を使用します（MCU向けは --disable-sampling 相当で
 *       APIを使わない運用を推奨）。
 */
int npycrf_decode_sample(const npycrf_model_t *model,
                         const uint8_t *utf8, size_t len,
                         npycrf_work_t *work,
                         void *sample_buf, size_t sample_buf_size,
                         double temperature,
                         uint32_t seed,
                         uint16_t *out_b_cp, size_t out_b_cap,
                         size_t *out_b_count,
                         npycrf_score_t *out_sample_score);

/*
 * 追加ワークバッファサイズ（N-best Viterbi 用）
 */
size_t npycrf_nbestbuf_size(uint16_t max_n_cp, uint16_t max_word_len, uint16_t nbest);

/*
 * N-best Viterbi デコード
 *
 * 出力はフラットな境界配列として返します。
 *  - out_b_cp_flat は [nbest][out_b_cap] 相当の領域（nbest*out_b_cap）を用意
 *  - out_b_count[i] に i番目候補の境界数を格納（0なら無効）
 *
 * @return >=0: 実際に出力された候補数, <0: エラー
 */
int npycrf_decode_nbest(const npycrf_model_t *model,
                        const uint8_t *utf8, size_t len,
                        npycrf_work_t *work,
                        void *nbest_buf, size_t nbest_buf_size,
                        uint16_t nbest,
                        uint16_t *out_b_cp_flat, size_t out_b_cap,
                        size_t *out_b_count,
                        npycrf_score_t *out_scores);

#ifdef __cplusplus
} /* extern "C" */
#endif
