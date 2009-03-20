/*
 * double_array_trie.c
 *
 * コンパクトなダブル配列トライ (BASE/CHECK) の C 実装
 *
 * - キーはバイト列（UTF-8 はバイト列として扱う）
 * - check==0 を「空きスロット」として使用
 * - キー終端遷移として明示的にコード 0 を使用
 * - 動的アロケーション (malloc) または静的バッファの両方に対応
 */

#include "double_array_trie.h"

#include <string.h>

#ifndef DA_NO_MALLOC
#include <stdlib.h>
#endif

/* ルートノードのインデックス。インデックス 0 は未使用 */
#define DA_ROOT 1

/* ---------------- 内部ヘルパー関数 ---------------- */

/* トライが初期化済みかチェック */
static int da_is_initialized(const da_trie_t *da) {
    return da && da->base && da->check && da->capacity > (size_t)DA_ROOT;
}

/* 必要な容量を確保（動的モードの場合は拡張） */
static int da_reserve(da_trie_t *da, size_t need) {
    if (!da || !da->base || !da->check) return DA_ERR_BADARG;
    if (need <= da->capacity) return DA_OK;
    if (!da->dynamic) return DA_ERR_FULL;

#ifdef DA_NO_MALLOC
    (void)need;
    return DA_ERR_FULL;
#else
    /* 2 の累乗で拡張 */
    size_t newcap = (da->capacity == 0) ? 256u : da->capacity;
    while (newcap < need) {
        size_t prev = newcap;
        newcap *= 2u;
        if (newcap < prev) {
            /* オーバーフロー */
            return DA_ERR_NOMEM;
        }
    }

    /*
     * 重要: base/check の拡張は「すべて成功」または「何もしない」トランザクションで行う
     * 2 回の realloc() を使うと、片方が失敗した場合にトライが不整合状態になる可能性がある
     */
    da_index_t *nb = (da_index_t *)calloc(newcap, sizeof(da_index_t));
    da_index_t *nc = (da_index_t *)calloc(newcap, sizeof(da_index_t));
    if (!nb || !nc) {
        free(nb);
        free(nc);
        return DA_ERR_NOMEM;
    }

    memcpy(nb, da->base, da->capacity * sizeof(da_index_t));
    memcpy(nc, da->check, da->capacity * sizeof(da_index_t));

    free(da->base);
    free(da->check);

    da->base = nb;
    da->check = nc;
    da->capacity = newcap;
    return DA_OK;
#endif
}

/* 状態遷移: cur から code で遷移した次のノードを返す */
static da_index_t da_next(const da_trie_t *da, da_index_t cur, uint8_t code) {
    if (!da_is_initialized(da)) return 0;
    if (cur <= 0 || (size_t)cur >= da->capacity) return 0;

    da_index_t b = da->base[cur];
    if (b <= 0) return 0;

    size_t idx = (size_t)b + (size_t)code;
    if (idx >= da->capacity) return 0;

    /*
     * 重要:
     *  - ルート(DA_ROOT=1)は check[DA_ROOT]=DA_ROOT として初期化される。
     *  - base[DA_ROOT] が 1 のとき、code=0 だと idx==cur になり得る。
     *    これを「自己ループ遷移」として扱うと、子コード収集や再配置で
     *    ルート自身を誤って子ノードとして扱い、木構造が破壊される。
     */
    if (idx == (size_t)cur) return 0;
    if (da->check[idx] == cur) return (da_index_t)idx;
    return 0;
}

/* 親ノードの既存の子コードを codes_out[] に収集。件数（<=256）を返す */
static size_t da_collect_children_codes(const da_trie_t *da, da_index_t parent, uint8_t codes_out[DA_ALPHABET_SIZE]) {
    size_t n = 0;
    da_index_t b = da->base[parent];
    if (b <= 0) return 0;

    for (int c = 0; c < DA_ALPHABET_SIZE; c++) {
        size_t idx = (size_t)b + (size_t)(uint8_t)c;
        /* parent 自身を子ノードとして数えてはいけない（ルートの自己ループ防止） */
        if (idx == (size_t)parent) continue;
        if (idx < da->capacity && da->check[idx] == parent) {
            codes_out[n++] = (uint8_t)c;
        }
    }
    return n;
}

/* コード配列に特定のコードが含まれるかチェック */
static int da_codes_contains(const uint8_t *codes, size_t n, uint8_t code) {
    for (size_t i = 0; i < n; i++) {
        if (codes[i] == code) return 1;
    }
    return 0;
}

/*
 * すべての codes[i] に対して、スロット (b+codes[i]) が利用可能な base 値 b を探す
 * この親が既に占有しているスロットは「利用可能」として扱う（再配置の可能性があるため）
 */
static int da_find_base(da_trie_t *da, da_index_t parent, const uint8_t *codes, size_t n, da_index_t *out_base) {
    if (!da || !codes || n == 0 || !out_base) return DA_ERR_BADARG;

    /* 使用される最大コード（容量チェック用） */
    uint8_t maxc = 0;
    for (size_t i = 0; i < n; i++) {
        if (codes[i] > maxc) maxc = codes[i];
    }

    /* 1 からスキャン: 低いインデックスを詰めた方が RAM/ROM に有利 */
    for (da_index_t b = 1; b > 0; b++) {
        size_t need = (size_t)b + (size_t)maxc + 1u;
        int rc = da_reserve(da, need);
        if (rc != DA_OK) return rc;

        int ok = 1;
        for (size_t i = 0; i < n; i++) {
            size_t idx = (size_t)b + (size_t)codes[i];
            /* 親ノード自身のスロットを子に割り当てるのは禁止 */
            if (idx == (size_t)parent) { ok = 0; break; }
            da_index_t chk = da->check[idx];
            if (chk != 0 && chk != parent) {
                ok = 0;
                break;
            }
        }
        if (ok) {
            *out_base = b;
            return DA_OK;
        }
    }

    return DA_ERR_FULL;
}

/* 親の既存の子をすべて old_base から new_base に再配置 */
static int da_relocate_children(da_trie_t *da, da_index_t parent, da_index_t new_base) {
    uint8_t codes[DA_ALPHABET_SIZE];
    size_t n = da_collect_children_codes(da, parent, codes);

    da_index_t old_base = da->base[parent];

    /* 重複を安全に移動するための状態保存 */
    da_index_t child_base[DA_ALPHABET_SIZE];
    da_index_t old_idx[DA_ALPHABET_SIZE];
    da_index_t new_idx[DA_ALPHABET_SIZE];

    for (size_t i = 0; i < n; i++) {
        size_t o = (size_t)old_base + (size_t)codes[i];
        size_t nn = (size_t)new_base + (size_t)codes[i];

        int rc = da_reserve(da, nn + 1u);
        if (rc != DA_OK) return rc;

        old_idx[i] = (da_index_t)o;
        new_idx[i] = (da_index_t)nn;
        child_base[i] = da->base[o];
    }

    /* 旧スロットをクリア */
    for (size_t i = 0; i < n; i++) {
        size_t o = (size_t)old_idx[i];
        da->base[o] = 0;
        da->check[o] = 0;
    }

    /* 新スロットに書き込み */
    for (size_t i = 0; i < n; i++) {
        size_t nn = (size_t)new_idx[i];
        da->check[nn] = parent;
        da->base[nn] = child_base[i];
    }

    /*
     * 孫ノードの check ポインタを修正 (old_child -> new_child)
     *
     * 注意: 1 回の再配置で複数の子ノードを動かすと、
     *       new_child が別の old_child と同じ値になることがある。
     *       その場合、逐次更新すると「更新済みの check」が別 old_child と誤一致し、
     *       不正な二重変換が起こる。
     *
     * 対策: まず一時的に負数でマーキングし（-new_child）、
     *       全マッピング適用後に負数を正に戻す。
     */
    for (size_t i = 0; i < n; i++) {
        da_index_t b = child_base[i];
        if (b <= 0) continue;
        da_index_t old_child = old_idx[i];
        da_index_t new_child = new_idx[i];
        for (int c = 0; c < DA_ALPHABET_SIZE; c++) {
            size_t g = (size_t)b + (size_t)(uint8_t)c;
            if (g < da->capacity && da->check[g] == old_child) {
                da->check[g] = (da_index_t)(-new_child);
            }
        }
    }

    /* 2nd pass: 負数を正のインデックスに戻す */
    for (size_t i = 0; i < n; i++) {
        da_index_t b = child_base[i];
        if (b <= 0) continue;
        da_index_t new_child = new_idx[i];
        da_index_t neg = (da_index_t)(-new_child);
        for (int c = 0; c < DA_ALPHABET_SIZE; c++) {
            size_t g = (size_t)b + (size_t)(uint8_t)c;
            if (g < da->capacity && da->check[g] == neg) {
                da->check[g] = new_child;
            }
        }
    }

    da->base[parent] = new_base;
    return DA_OK;
}

/* parent --code--> の遷移が存在することを保証 */
static int da_ensure_transition(da_trie_t *da, da_index_t parent, uint8_t code, da_index_t *out_next) {
    if (!da || !out_next) return DA_ERR_BADARG;
    if (!da_is_initialized(da)) return DA_ERR_BADARG;
    if (parent <= 0 || (size_t)parent >= da->capacity) return DA_ERR_BADARG;

    da_index_t b = da->base[parent];
    if (b <= 0) {
        uint8_t one[1] = { code };
        da_index_t new_base = 0;
        int rc = da_find_base(da, parent, one, 1u, &new_base);
        if (rc != DA_OK) return rc;
        da->base[parent] = new_base;
        b = new_base;
    }

    size_t idx = (size_t)b + (size_t)code;
    int rc = da_reserve(da, idx + 1u);
    if (rc != DA_OK) return rc;

    da_index_t chk = da->check[idx];
    if (chk == parent) {
        *out_next = (da_index_t)idx;
        return DA_OK;
    }
    if (chk == 0) {
        da->check[idx] = parent;
        da->base[idx] = 0;
        *out_next = (da_index_t)idx;
        return DA_OK;
    }

    /* 衝突: 既存の子 + この新コードを再配置 */
    uint8_t codes[DA_ALPHABET_SIZE];
    size_t n = da_collect_children_codes(da, parent, codes);
    if (!da_codes_contains(codes, n, code)) {
        codes[n++] = code;
    }

    da_index_t new_base = 0;
    rc = da_find_base(da, parent, codes, n, &new_base);
    if (rc != DA_OK) return rc;

    rc = da_relocate_children(da, parent, new_base);
    if (rc != DA_OK) return rc;

    idx = (size_t)new_base + (size_t)code;
    rc = da_reserve(da, idx + 1u);
    if (rc != DA_OK) return rc;

    if (da->check[idx] != 0) {
        /* 本来起こらないはず */
        return DA_ERR_FULL;
    }

    da->check[idx] = parent;
    da->base[idx] = 0;
    *out_next = (da_index_t)idx;
    return DA_OK;
}

/* ---------------- 公開 API ---------------- */

int da_trie_init_dynamic(da_trie_t *da, size_t initial_capacity) {
    if (!da) return DA_ERR_BADARG;
    memset(da, 0, sizeof(*da));

#ifdef DA_NO_MALLOC
    (void)initial_capacity;
    return DA_ERR_FULL;
#else
    if (initial_capacity < 16u) initial_capacity = 16u;

    da->base  = (da_index_t *)calloc(initial_capacity, sizeof(da_index_t));
    da->check = (da_index_t *)calloc(initial_capacity, sizeof(da_index_t));
    if (!da->base || !da->check) {
        free(da->base);
        free(da->check);
        memset(da, 0, sizeof(*da));
        return DA_ERR_NOMEM;
    }

    da->capacity = initial_capacity;
    da->dynamic = 1;

    return da_trie_clear(da);
#endif
}

int da_trie_init_static(da_trie_t *da, da_index_t *base, da_index_t *check, size_t capacity) {
    if (!da || !base || !check) return DA_ERR_BADARG;
    if (capacity < 16u) return DA_ERR_BADARG;

    memset(da, 0, sizeof(*da));
    da->base = base;
    da->check = check;
    da->capacity = capacity;
    da->dynamic = 0;

    return da_trie_clear(da);
}

void da_trie_free(da_trie_t *da) {
    if (!da) return;

#ifndef DA_NO_MALLOC
    if (da->dynamic) {
        free(da->base);
        free(da->check);
    }
#endif

    memset(da, 0, sizeof(*da));
}

int da_trie_clear(da_trie_t *da) {
    if (!da || !da->base || !da->check || da->capacity < 2u) return DA_ERR_BADARG;

    memset(da->base, 0, da->capacity * sizeof(da_index_t));
    memset(da->check, 0, da->capacity * sizeof(da_index_t));

    /* ルートは占有済みにする */
    da->base[DA_ROOT] = 1;
    da->check[DA_ROOT] = DA_ROOT;

    return DA_OK;
}

int da_trie_add_utf8(da_trie_t *da, const char *utf8) {
    if (!da || !utf8) return DA_ERR_BADARG;
    return da_trie_add_bytes(da, (const uint8_t *)utf8, strlen(utf8));
}

int da_trie_add_bytes(da_trie_t *da, const uint8_t *bytes, size_t len) {
    if (!da || !bytes) return DA_ERR_BADARG;
    /* 空キー（長さ0）はルート自己ループと衝突しやすく、サポートしない */
    if (len == 0) return DA_ERR_BADARG;
    if (!da_is_initialized(da)) return DA_ERR_BADARG;

    da_index_t cur = DA_ROOT;
    for (size_t i = 0; i < len; i++) {
        da_index_t next = 0;
        int rc = da_ensure_transition(da, cur, bytes[i], &next);
        if (rc != DA_OK) return rc;
        cur = next;
    }

    /* キー終端マーカー (バイト 0) */
    {
        da_index_t term = 0;
        int rc = da_ensure_transition(da, cur, 0u, &term);
        if (rc != DA_OK) return rc;
    }

    return DA_OK;
}

int da_trie_contains_utf8(const da_trie_t *da, const char *utf8) {
    if (!da || !utf8) return 0;
    return da_trie_contains_bytes(da, (const uint8_t *)utf8, strlen(utf8));
}

int da_trie_contains_bytes(const da_trie_t *da, const uint8_t *bytes, size_t len) {
    if (!da || !bytes) return 0;
    if (!da_is_initialized(da)) return 0;

    da_index_t cur = DA_ROOT;
    for (size_t i = 0; i < len; i++) {
        cur = da_next(da, cur, bytes[i]);
        if (cur == 0) return 0;
    }

    return da_next(da, cur, 0u) ? 1 : 0;
}

da_index_t da_trie_search_prefix_bytes(const da_trie_t *da, const uint8_t *bytes, size_t len) {
    if (!da || !bytes) return 0;
    if (!da_is_initialized(da)) return 0;

    da_index_t cur = DA_ROOT;
    for (size_t i = 0; i < len; i++) {
        cur = da_next(da, cur, bytes[i]);
        if (cur == 0) return 0;
    }
    return cur;
}

/* ---------------- 読み取り専用ビュー API ---------------- */

static int da_ro_is_initialized(const da_trie_ro_t *da) {
    return da && da->base && da->check && da->capacity > (size_t)DA_ROOT;
}

static da_index_t da_next_ro(const da_trie_ro_t *da, da_index_t cur, uint8_t code) {
    if (!da_ro_is_initialized(da)) return 0;
    if (cur <= 0 || (size_t)cur >= da->capacity) return 0;

    da_index_t b = da->base[cur];
    if (b <= 0) return 0;

    size_t idx = (size_t)b + (size_t)code;
    if (idx >= da->capacity) return 0;

    /* ルート自己ループを遷移として扱わない（動的版と同様） */
    if (idx == (size_t)cur) return 0;
    if (da->check[idx] == cur) return (da_index_t)idx;
    return 0;
}

int da_trie_ro_contains_utf8(const da_trie_ro_t *da, const char *utf8) {
    if (!da || !utf8) return 0;
    return da_trie_ro_contains_bytes(da, (const uint8_t *)utf8, strlen(utf8));
}

int da_trie_ro_contains_bytes(const da_trie_ro_t *da, const uint8_t *bytes, size_t len) {
    if (!da || !bytes) return 0;
    if (!da_ro_is_initialized(da)) return 0;

    da_index_t cur = DA_ROOT;
    for (size_t i = 0; i < len; i++) {
        cur = da_next_ro(da, cur, bytes[i]);
        if (cur == 0) return 0;
    }
    return da_next_ro(da, cur, 0u) ? 1 : 0;
}

da_index_t da_trie_ro_search_prefix_bytes(const da_trie_ro_t *da, const uint8_t *bytes, size_t len) {
    if (!da || !bytes) return 0;
    if (!da_ro_is_initialized(da)) return 0;

    da_index_t cur = DA_ROOT;
    for (size_t i = 0; i < len; i++) {
        cur = da_next_ro(da, cur, bytes[i]);
        if (cur == 0) return 0;
    }
    return cur;
}
