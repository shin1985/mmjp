#pragma once

/*
 * mmjp_model.h
 *
 * 目的:
 *  - NPYCRF Lite が推論に必要とするパラメータ（辞書トライ + LM + CRF）を
 *    まとめて扱うための薄いI/O層。
 *
 * 方針:
 *  - MCU では「バイナリロード」よりも「C配列としてflash/ROMに埋め込み」が扱いやすい。
 *  - CLI では学習→保存→テストを繰り返すため、簡易なバイナリ形式を用意する。
 *
 * 注意:
 *  - このヘッダのバイナリ形式は、同梱ツール用の簡易フォーマットです。
 *    長期互換を保証するものではないので version を確認してください。
 */

#include <stdint.h>
#include <stddef.h>

#include "../npycrf_lite/npycrf_lite.h"

#ifdef __cplusplus
extern "C" {
#endif

/* v1: 初期フォーマット（日本語ハードコード） */
#define MMJP_MODEL_MAGIC_V1 "MMJPv1\0\0" /* 8 bytes */
#define MMJP_MODEL_VERSION_V1 1u

/* v2: 言語非依存化（flags, cc_mode, cc_ranges） */
#define MMJP_MODEL_MAGIC "MMJPv2\0\0" /* 8 bytes */
#define MMJP_MODEL_VERSION 2u

typedef struct {
  npycrf_model_t m;

  /* cc_ranges の所有バッファ（動的確保した場合） */
  npycrf_cc_range_t *cc_ranges_owned;
  uint32_t cc_ranges_count;

  /* 所有バッファ（free 対象）。load した場合のみ非NULL。 */
  void *owned;
  size_t owned_bytes;
} mmjp_loaded_model_t;

/*
 * バイナリ保存（CLI用）
 *
 * m->lm.trie は ro_view なので、base/check 実体は呼び出し側が持っている前提。
 * この関数は m 内のポインタが指す配列をそのまま書き出します。
 */
int mmjp_model_save_bin(const char *path, const npycrf_model_t *m);

/*
 * バイナリ読み込み（CLI用）
 *
 *  - out->m のポインタは読み込んだ owned バッファを指します。
 *  - 使い終わったら mmjp_model_free() を呼んでください。
 */
int mmjp_model_load_bin(const char *path, mmjp_loaded_model_t *out);
void mmjp_model_free(mmjp_loaded_model_t *m);

#ifdef __cplusplus
}
#endif
