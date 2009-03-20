/*
 * sn_suffix_array_utf8.h
 *
 * 旧 SN_SUFFIX_ARRAY API に似た薄い互換ラッパー
 * プレーン C かつ UTF-8 対応
 *
 * 特徴:
 *   - バッファを渡せばヒープ不要
 *   - ファイル I/O はオプション（SN_SA_ENABLE_FILEIO=1 を定義）
 */

#ifndef SN_SUFFIX_ARRAY_UTF8_H
#define SN_SUFFIX_ARRAY_UTF8_H

#include <stddef.h>
#include <stdint.h>

#include "sa_utf8.h"

#ifdef __cplusplus
extern "C" {
#endif

/* 接尾辞配列オブジェクト構造体 */
typedef struct {
  char name[128];             /* 名前（識別用） */
  const uint8_t *text;        /* テキストバッファ（所有しない） */
  size_t text_len;            /* テキストのバイト長 */
  sa_idx_t *sa;               /* 接尾辞配列バッファ */
  size_t sa_len;              /* 接尾辞数 */
  size_t sa_cap;              /* バッファ容量 */
  unsigned build_flags;       /* 構築フラグ */
} sn_suffix_array_t;

/*
 * メモリ上のテキストバッファから初期化
 * テキストバッファはコピーされない（呼び出し側が保持する必要あり）
 *
 * 戻り値: 成功時 1、失敗時 0
 */
int sn_sa_build(sn_suffix_array_t *obj,
                const char *name,
                const uint8_t *text, size_t text_len,
                sa_idx_t *sa_buf, size_t sa_buf_cap,
                unsigned build_flags);

/*
 * キーワードの出現数をカウント
 * keyword は NUL 終端 UTF-8 文字列
 */
size_t sn_sa_get_count(const sn_suffix_array_t *obj, const char *keyword);

/*
 * バイグラムカウント: forward の出現数、および forward+back の連続出現数
 */
sa_bigram_count_t sn_sa_get_bigram_count(const sn_suffix_array_t *obj,
                                         const char *forward_word,
                                         const char *back_word);

/*
 * デバッグ用ヘルパー: 各接尾辞の先頭 n_codepoints とそのカウントを表示
 */
void sn_sa_show_ngram(const sn_suffix_array_t *obj, size_t n_codepoints);

#ifdef SN_SA_ENABLE_FILEIO
/*
 * オプション: 保存/読み込み
 * シンプルなバイナリ形式（オリジナルの 2008 年版ファイルとは非互換）
 */
int sn_sa_save(const sn_suffix_array_t *obj, const char *path);
int sn_sa_load(sn_suffix_array_t *obj,
               const char *path,
               uint8_t *text_buf, size_t text_buf_cap,
               sa_idx_t *sa_buf, size_t sa_buf_cap,
               unsigned build_flags);
#endif

#ifdef __cplusplus
}
#endif

#endif /* SN_SUFFIX_ARRAY_UTF8_H */
