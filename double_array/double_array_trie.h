/*
 * double_array_trie.h
 *
 * ダブル配列トライ (BASE/CHECK) の C99 実装
 *
 * UTF-8 サポート:
 *   - 入力文字列を UTF-8 *バイト列* として扱う
 *   - 各遷移は 1 バイト (0..255)。バイト 0 はキー終端マーカーとして予約
 *   - 非 ASCII 文字（例: 日本語）は複数ステップを使用（1 コードポイントあたり 3 バイト）
 *
 * マイコン向け設計:
 *   - 静的/malloc 不使用モード対応（呼び出し側でバッファを提供）
 *   - インデックス型は設定可能（例: int16_t で RAM 節約）
 *   - wchar_t/ロケール/ワイド I/O は使用しない
 *
 * 注意:
 *   - 古い C++ ヘッダオンリー実装をリファクタリングして C に移植したもの
 */

#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * インデックス型の設定
 * 例（RAM 節約、ただしノード数上限が小さくなる）:
 *   -DDA_INDEX_T=int16_t
 */
#ifndef DA_INDEX_T
#define DA_INDEX_T int32_t
#endif

typedef DA_INDEX_T da_index_t;

/* アルファベットサイズ: バイト (0..255) */
#ifndef DA_ALPHABET_SIZE
#define DA_ALPHABET_SIZE 256
#endif

/* エラーコード */
enum {
    DA_OK = 0,           /* 成功 */
    DA_ERR_BADARG = -1,  /* 不正な引数 */
    DA_ERR_NOMEM  = -2,  /* メモリ確保失敗 */
    DA_ERR_FULL   = -3,  /* 容量不足 */
};

/* ダブル配列トライ構造体 */
typedef struct {
    da_index_t *base;    /* BASE 配列 */
    da_index_t *check;   /* CHECK 配列 */
    size_t capacity;     /* エントリ数（容量） */

    /* 非ゼロの場合、このインスタンスがメモリを所有し、malloc/calloc で拡張可能 */
    int dynamic;
} da_trie_t;

/* 読み取り専用ビュー（base/check が flash/ROM に const として格納されている場合に便利） */
typedef struct {
    const da_index_t *base;
    const da_index_t *check;
    size_t capacity;
} da_trie_ro_t;

/* 読み取り専用検索 API */
int da_trie_ro_contains_utf8(const da_trie_ro_t *da, const char *utf8);
int da_trie_ro_contains_bytes(const da_trie_ro_t *da, const uint8_t *bytes, size_t len);
da_index_t da_trie_ro_search_prefix_bytes(const da_trie_ro_t *da, const uint8_t *bytes, size_t len);

/* 動的アロケーション (malloc/calloc) で初期化 */
int da_trie_init_dynamic(da_trie_t *da, size_t initial_capacity);

/* 呼び出し側提供のバッファで初期化（malloc 不使用）。バッファはゼロクリアされる */
int da_trie_init_static(da_trie_t *da,
                        da_index_t *base,
                        da_index_t *check,
                        size_t capacity);

/* 内部バッファを解放（動的モードのみ） */
void da_trie_free(da_trie_t *da);

/* トライの内容をクリア（バッファは保持） */
int da_trie_clear(da_trie_t *da);

/* キーを挿入 */
int da_trie_add_utf8(da_trie_t *da, const char *utf8);
int da_trie_add_bytes(da_trie_t *da, const uint8_t *bytes, size_t len);

/* 完全キー検索。存在すれば 1、なければ 0 を返す */
int da_trie_contains_utf8(const da_trie_t *da, const char *utf8);
int da_trie_contains_bytes(const da_trie_t *da, const uint8_t *bytes, size_t len);

/*
 * プレフィックス走査（旧 search_str 相当）:
 * len バイトを消費した後のノードインデックスを返す。パスが存在しなければ 0
 */
da_index_t da_trie_search_prefix_bytes(const da_trie_t *da, const uint8_t *bytes, size_t len);

#ifdef __cplusplus
}
#endif
