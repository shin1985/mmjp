/*
 * sa_utf8.h
 *
 * メモリ制約のあるターゲット向け UTF-8 対応（コードポイント境界）接尾辞配列
 *
 * 特徴:
 *   - テキスト自体は生の UTF-8 バイトとして扱う
 *   - 接尾辞の開始位置は UTF-8 コードポイント境界のみで生成（継続バイトでは生成しない）
 *   - オプションフィルタリング: ASCII 空白および/または ASCII 句読点を接尾辞開始位置からスキップ可能
 *   - ヒープアロケーション不要: 呼び出し側が接尾辞配列バッファを提供
 *
 * この実装は提供された 2008 年頃のコードをリファクタリング/移植したもの:
 *   - 固定長処理（NUL 終端バッファに依存しない）
 *   - UCHAR_MAX オフバイワンバグを修正（該当ステージを削除）
 *   - LCP 依存を削除（RAM 節約）、代わりに 2 回の二分探索境界を使用
 *   - 反復的 3-way 基数クイックソート（明示的な小さいスタック）で再帰を回避
 */

#ifndef SA_UTF8_H
#define SA_UTF8_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* インデックス型（設定可能） */
#ifndef SA_IDX_T
#define SA_IDX_T uint32_t
#endif

typedef SA_IDX_T sa_idx_t;

/* 接尾辞配列ビュー構造体 */
typedef struct {
  const uint8_t *text;   /* UTF-8 バイト文字列 */
  size_t text_len;       /* テキストのバイト数 */
  const sa_idx_t *sa;    /* 接尾辞配列: テキストへのバイトオフセット */
  size_t sa_len;         /* 接尾辞（開始位置）の数 */
} sa_utf8_view_t;

/* バイグラムカウント結果 */
typedef struct {
  size_t forward;        /* 前方語の出現数 */
  size_t forward_back;   /* 前方語 + 後方語の連続出現数 */
} sa_bigram_count_t;

/* 構築フラグ */
enum sa_build_flags {
  SA_BUILD_DEFAULT            = 0u,
  SA_BUILD_SKIP_ASCII_SPACE   = 1u << 0,  /* ASCII 空白位置をスキップ */
  SA_BUILD_SKIP_ASCII_PUNCT   = 1u << 1,  /* ASCII 句読点位置をスキップ */
  SA_BUILD_VALIDATE_UTF8      = 1u << 2   /* 強い境界検証（遅い） */
};

/* 指定フラグでこのテキストに対して生成される接尾辞開始位置の数を返す */
size_t sa_utf8_count_starts(const uint8_t *text, size_t text_len, unsigned flags);

/*
 * 接尾辞配列を sa_out（容量 sa_out_cap）に構築。書き込んだ接尾辞数を返す。
 * エラー時（容量不足、NULL 入力など）は 0 を返す。
 */
size_t sa_utf8_build(sa_idx_t *sa_out, size_t sa_out_cap,
                     const uint8_t *text, size_t text_len,
                     unsigned flags);

/* 既に構築済みの接尾辞配列からビューを作成 */
static inline sa_utf8_view_t sa_utf8_view(const uint8_t *text, size_t text_len,
                                         const sa_idx_t *sa, size_t sa_len) {
  sa_utf8_view_t v;
  v.text = text;
  v.text_len = text_len;
  v.sa = sa;
  v.sa_len = sa_len;
  return v;
}

/* key が任意の接尾辞のプレフィックスとして出現する回数をカウント（テキスト内の出現数） */
size_t sa_utf8_count_prefix(const sa_utf8_view_t *view,
                            const uint8_t *key, size_t key_len);

/* バイグラム出現数をカウント: forward の出現数、および forward の直後に back が続く出現数 */
sa_bigram_count_t sa_utf8_count_bigram(const sa_utf8_view_t *view,
                                       const uint8_t *forward, size_t forward_len,
                                       const uint8_t *back, size_t back_len);

/*
 * ユーティリティ: バイトオフセット start から始まる最初の n_codepoints 個の UTF-8 コードポイントを out にコピー
 * - out_cap > 0 の場合は常に NUL 終端
 * - 書き込んだバイト数（NUL を除く）を返す
 */
size_t sa_utf8_copy_prefix_n(const uint8_t *text, size_t text_len,
                             size_t start, size_t n_codepoints,
                             char *out, size_t out_cap,
                             unsigned flags);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* SA_UTF8_H */
