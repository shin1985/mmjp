/*
 * mmjp_lossless.h
 *
 * 可逆トークナイズ（Lossless Tokenization）
 *
 * 空白/タブ/改行をメタ文字に変換し、tokenize → detok で完全復元可能にする。
 * SentencePiece互換の思想で実装。
 *
 * メタ文字定義:
 *   ▀ (U+2580) - エスケープ文字
 *   ▁ (U+2581) - スペース
 *   ▂ (U+2582) - タブ
 *   ▃ (U+2583) - LF (改行)
 *   ▄ (U+2584) - CR
 *
 * 衝突回避:
 *   入力に ▀▁▂▃▄ が含まれる場合は、▀ + 元文字 にエスケープ
 */

#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* メタ文字コードポイント */
#define MMJP_LOSSLESS_ESCAPE 0x2580u  /* ▀ */
#define MMJP_LOSSLESS_SPACE  0x2581u  /* ▁ */
#define MMJP_LOSSLESS_TAB    0x2582u  /* ▂ */
#define MMJP_LOSSLESS_LF     0x2583u  /* ▃ */
#define MMJP_LOSSLESS_CR     0x2584u  /* ▄ */

/*
 * losslessエンコード
 *
 * 通常テキストをlossless形式に変換する。
 *
 * @param src            入力UTF-8文字列
 * @param src_len        入力バイト長
 * @param dst            出力バッファ（NULLの場合は必要サイズを計算）
 * @param dst_cap        出力バッファ容量
 * @param include_newlines  1の場合、LF/CRも変換（--read_all用）
 * @return 出力バイト数（NUL終端を含まない）、エラー時は0
 *
 * 変換規則:
 *   ' '  → ▁
 *   '\t' → ▂
 *   '\n' → ▃ (include_newlines=1の場合のみ)
 *   '\r' → ▄ (include_newlines=1の場合のみ)
 *   ▀▁▂▃▄ → ▀ + 元文字 (エスケープ)
 */
size_t mmjp_lossless_encode(const uint8_t *src, size_t src_len,
                            uint8_t *dst, size_t dst_cap,
                            int include_newlines);

/*
 * losslessデコード
 *
 * lossless形式を元のテキストに復元する。
 *
 * @param src      入力UTF-8文字列（lossless形式）
 * @param src_len  入力バイト長
 * @param dst      出力バッファ（NULLの場合は必要サイズを計算）
 * @param dst_cap  出力バッファ容量
 * @return 出力バイト数（NUL終端を含まない）、エラー時は0
 *
 * 変換規則（エンコードの逆）:
 *   ▁ → ' '
 *   ▂ → '\t'
 *   ▃ → '\n'
 *   ▄ → '\r'
 *   ▀X → X (エスケープ解除)
 */
size_t mmjp_lossless_decode(const uint8_t *src, size_t src_len,
                            uint8_t *dst, size_t dst_cap);

/*
 * detokenize用ヘルパー
 *
 * トークン列を結合してデコードする。
 * トークン間の区切りスペースは無視される。
 *
 * @param tokens       トークン配列（UTF-8文字列のポインタ）
 * @param token_lens   各トークンのバイト長
 * @param n_tokens     トークン数
 * @param dst          出力バッファ
 * @param dst_cap      出力バッファ容量
 * @return 出力バイト数、エラー時は0
 */
size_t mmjp_lossless_detokenize(const uint8_t **tokens, const size_t *token_lens,
                                size_t n_tokens,
                                uint8_t *dst, size_t dst_cap);

#ifdef __cplusplus
}
#endif
