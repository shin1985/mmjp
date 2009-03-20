/*
 * unilm_mdl.c
 *
 * ユニグラム言語モデル MDL 学習・推論の実装
 */

#include "unilm_mdl.h"

#include <string.h>
#include <math.h>

#ifndef UNILM_NO_MALLOC
#include <stdlib.h>
#endif

/* ダブル配列トライのルートノードインデックス */
#define UNILM_DA_ROOT 1

/* ---------------- UTF-8 ヘルパー関数 ---------------- */

/* 次の UTF-8 コードポイント境界（バイトオフセット）を返す */
static size_t unilm_utf8_next(const uint8_t *s, size_t len, size_t pos) {
  if (pos >= len) return len;
  uint8_t b = s[pos];
  if (b < 0x80u) return pos + 1u;
  if ((b & 0xE0u) == 0xC0u) {
    size_t n = pos + 2u;
    return (n > len) ? len : n;
  }
  if ((b & 0xF0u) == 0xE0u) {
    size_t n = pos + 3u;
    return (n > len) ? len : n;
  }
  if ((b & 0xF8u) == 0xF0u) {
    size_t n = pos + 4u;
    return (n > len) ? len : n;
  }
  /* 無効な先頭バイト: 1 バイトとして扱う */
  return pos + 1u;
}

/* UTF-8 コードポイント数をカウント */
static size_t unilm_utf8_count_codepoints(const uint8_t *s, size_t len) {
  size_t n = 0;
  for (size_t pos = 0; pos < len; ) {
    size_t next = unilm_utf8_next(s, len, pos);
    if (next <= pos) break;
    pos = next;
    n++;
  }
  return n;
}

/*
 * コードポイント境界オフセットを構築: cp_off[0]=0, cp_off[M]=len
 * UNILM_OK を返し、*out_m=M を設定
 */
static int unilm_build_cp_offsets(const uint8_t *s, size_t len,
                                 uint32_t *cp_off, size_t cp_off_cap,
                                 size_t *out_m) {
  if (!cp_off || cp_off_cap == 0 || !out_m) return UNILM_ERR_BADARG;
  size_t m = 0;
  cp_off[0] = 0;
  size_t pos = 0;
  while (pos < len) {
    if (m + 1u >= cp_off_cap) return UNILM_ERR_RANGE;
    size_t next = unilm_utf8_next(s, len, pos);
    if (next <= pos) return UNILM_ERR_UTF8;
    pos = next;
    m++;
    cp_off[m] = (uint32_t)pos;
  }
  *out_m = m;
  return UNILM_OK;
}

/* ---------------- 対数ドメインヘルパー ---------------- */

/* log(exp(a)+exp(b)) を計算 */
static inline unilm_real_t unilm_log_add(unilm_real_t a, unilm_real_t b) {
  if (!isfinite((double)a)) return b;
  if (!isfinite((double)b)) return a;
  if (a < b) {
    unilm_real_t t = a;
    a = b;
    b = t;
  }
  /* a >= b */
  unilm_real_t d = b - a;
  if (d < (unilm_real_t)-50) return a;
  return a + (unilm_real_t)log1p(exp((double)d));
}

/* ---------------- ダブル配列トライヘルパー ----------------
 *
 * ピース ID を終端ノードに以下の形式で格納:
 *   trie.base[term_node] = -(id+1)
 */

/* 生の遷移を実行 */
static inline da_index_t unilm_da_next_raw(const da_index_t *base,
                                          const da_index_t *check,
                                          size_t cap,
                                          da_index_t cur,
                                          uint8_t code) {
  if (!base || !check) return 0;
  if (cur <= 0 || (size_t)cur >= cap) return 0;
  da_index_t b = base[cur];
  if (b <= 0) return 0;
  size_t idx = (size_t)b + (size_t)code;
  if (idx >= cap) return 0;
  if (check[idx] == cur) return (da_index_t)idx;
  return 0;
}

/* 終端ノードから ID を取得 */
static int32_t unilm_da_term_id(const da_trie_t *da, da_index_t node) {
  if (!da || !da->base || !da->check) return -1;
  da_index_t term = unilm_da_next_raw(da->base, da->check, da->capacity, node, 0u);
  if (term == 0) return -1;
  da_index_t v = da->base[term];
  if (v >= 0) return -1;
  return (int32_t)(-v - 1);
}

/* 終端ノードに ID を設定 */
static int unilm_da_set_term_id(da_trie_t *da, const uint8_t *bytes, size_t len, uint32_t id) {
  if (!da || !da->base || !da->check || !bytes || len == 0) return UNILM_ERR_BADARG;
  da_index_t cur = UNILM_DA_ROOT;
  for (size_t i = 0; i < len; i++) {
    cur = unilm_da_next_raw(da->base, da->check, da->capacity, cur, bytes[i]);
    if (cur == 0) return UNILM_ERR_INTERNAL;
  }
  da_index_t term = unilm_da_next_raw(da->base, da->check, da->capacity, cur, 0u);
  if (term == 0) return UNILM_ERR_INTERNAL;
  /* -(id+1) を格納 */
  da->base[term] = (da_index_t)(-( (da_index_t)id + 1 ));
  return UNILM_OK;
}

/* ---------------- コーパスヘルパー ---------------- */

/* 配列コーパスイテレータの next 関数 */
int unilm_array_corpus_next(void *user, const uint8_t **out, size_t *out_len) {
  unilm_array_corpus_t *st = (unilm_array_corpus_t *)user;
  if (!st || !out || !out_len) return -1;
  if (st->i >= st->n) return 0;
  *out = st->sent[st->i];
  *out_len = st->sent_len[st->i];
  st->i++;
  return 1;
}

/* 配列コーパスイテレータの reset 関数 */
void unilm_array_corpus_reset(void *user) {
  unilm_array_corpus_t *st = (unilm_array_corpus_t *)user;
  if (st) st->i = 0;
}

/* ---------------- モデル ---------------- */

/* モデルが初期化済みかチェック */
static int unilm_model_is_init(const unilm_model_t *m) {
  return m && m->strbuf && m->pieces && m->logp && m->vocab_cap > 0 && m->strbuf_cap > 0 && m->trie.base && m->trie.check;
}

#ifndef UNILM_NO_MALLOC

/* 動的アロケーションでモデルを初期化 */
int unilm_model_init_dynamic(unilm_model_t *m,
                            size_t vocab_cap,
                            size_t strbuf_cap,
                            size_t da_cap) {
  if (!m || vocab_cap == 0 || strbuf_cap == 0 || da_cap < 16u) return UNILM_ERR_BADARG;
  memset(m, 0, sizeof(*m));

  m->strbuf = (uint8_t *)malloc(strbuf_cap);
  m->pieces = (unilm_piece_t *)calloc(vocab_cap, sizeof(unilm_piece_t));
  m->logp = (unilm_real_t *)calloc(vocab_cap, sizeof(unilm_real_t));
  if (!m->strbuf || !m->pieces || !m->logp) {
    unilm_model_free(m);
    return UNILM_ERR_NOMEM;
  }
  m->strbuf_cap = strbuf_cap;
  m->vocab_cap = vocab_cap;
  m->dynamic = 1;

  if (da_trie_init_dynamic(&m->trie, da_cap) != DA_OK) {
    unilm_model_free(m);
    return UNILM_ERR_NOMEM;
  }
  return unilm_model_clear(m);
}

#else

int unilm_model_init_dynamic(unilm_model_t *m,
                            size_t vocab_cap,
                            size_t strbuf_cap,
                            size_t da_cap) {
  (void)m; (void)vocab_cap; (void)strbuf_cap; (void)da_cap;
  return UNILM_ERR_NOMEM;
}

#endif

/* 静的バッファでモデルを初期化 */
int unilm_model_init_static(unilm_model_t *m,
                           uint8_t *strbuf, size_t strbuf_cap,
                           unilm_piece_t *pieces, unilm_real_t *logp, size_t vocab_cap,
                           da_index_t *da_base, da_index_t *da_check, size_t da_cap) {
  if (!m || !strbuf || !pieces || !logp || !da_base || !da_check) return UNILM_ERR_BADARG;
  if (strbuf_cap == 0 || vocab_cap == 0 || da_cap < 16u) return UNILM_ERR_BADARG;
  memset(m, 0, sizeof(*m));

  m->strbuf = strbuf;
  m->strbuf_cap = strbuf_cap;
  m->pieces = pieces;
  m->logp = logp;
  m->vocab_cap = vocab_cap;
  m->dynamic = 0;

  if (da_trie_init_static(&m->trie, da_base, da_check, da_cap) != DA_OK) {
    return UNILM_ERR_BADARG;
  }
  return unilm_model_clear(m);
}

/* モデルを解放 */
void unilm_model_free(unilm_model_t *m) {
  if (!m) return;
#ifndef UNILM_NO_MALLOC
  if (m->dynamic) {
    free(m->strbuf);
    free(m->pieces);
    free(m->logp);
    da_trie_free(&m->trie);
  }
#endif
  memset(m, 0, sizeof(*m));
}

/* モデルをクリア */
int unilm_model_clear(unilm_model_t *m) {
  if (!m || !m->strbuf || !m->pieces || !m->logp) return UNILM_ERR_BADARG;
  m->strbuf_len = 0;
  m->vocab_size = 0;
  if (da_trie_clear(&m->trie) != DA_OK) return UNILM_ERR_INTERNAL;
  return UNILM_OK;
}

/* 既存 ID を検索 */
int32_t unilm_model_find_id(const unilm_model_t *m,
                           const uint8_t *bytes, size_t len) {
  if (!unilm_model_is_init(m) || !bytes || len == 0) return -1;
  da_index_t cur = UNILM_DA_ROOT;
  for (size_t i = 0; i < len; i++) {
    cur = unilm_da_next_raw(m->trie.base, m->trie.check, m->trie.capacity, cur, bytes[i]);
    if (cur == 0) return -1;
  }
  return unilm_da_term_id(&m->trie, cur);
}

/* ピースのバイト列ポインタを取得 */
const uint8_t *unilm_model_piece_bytes(const unilm_model_t *m, size_t id, size_t *len_out) {
  if (!m || id >= m->vocab_size) return NULL;
  const unilm_piece_t *p = &m->pieces[id];
  if (len_out) *len_out = p->len;
  return m->strbuf + p->str_off;
}

/* ピースを追加 */
int32_t unilm_model_add_piece(unilm_model_t *m,
                             const uint8_t *bytes, size_t len,
                             uint8_t flags) {
  if (!unilm_model_is_init(m) || !bytes || len == 0) return -1;

  int32_t existing = unilm_model_find_id(m, bytes, len);
  if (existing >= 0) {
    m->pieces[existing].flags |= flags;
    return existing;
  }

  if (m->vocab_size >= m->vocab_cap) return -1;
  if (m->strbuf_len + len > m->strbuf_cap) return -1;

  uint32_t id = (uint32_t)m->vocab_size;
  uint32_t off = (uint32_t)m->strbuf_len;
  memcpy(m->strbuf + m->strbuf_len, bytes, len);
  m->strbuf_len += len;

  unilm_piece_t *p = &m->pieces[id];
  p->str_off = off;
  p->len = (uint16_t)len;
  p->len_cp = (uint16_t)unilm_utf8_count_codepoints(bytes, len);
  p->flags = flags;

  if (da_trie_add_bytes(&m->trie, bytes, len) != DA_OK) return -1;
  if (unilm_da_set_term_id(&m->trie, bytes, len, id) != UNILM_OK) return -1;

  /* 初期 logp は 0; 呼び出し側で正規化または EM を実行 */
  m->logp[id] = (unilm_real_t)0;
  m->vocab_size++;
  return (int32_t)id;
}

/* logp を設定 */
int unilm_model_set_logp(unilm_model_t *m, uint32_t id, unilm_real_t logp) {
  if (!m || id >= m->vocab_size) return UNILM_ERR_BADARG;
  m->logp[id] = logp;
  return UNILM_OK;
}

/* 確率を正規化 */
int unilm_model_normalize(unilm_model_t *m, unilm_real_t min_prob) {
  if (!m || m->vocab_size == 0) return UNILM_ERR_BADARG;
  if (!(min_prob > 0)) min_prob = (unilm_real_t)1e-12;

  /* logp->p に変換、正規化、フロア適用、再正規化 */
  unilm_real_t sum = 0;
  for (size_t i = 0; i < m->vocab_size; i++) {
    unilm_real_t p = (unilm_real_t)exp((double)m->logp[i]);
    if (!(p > 0)) p = 0;
    sum += p;
  }
  if (!(sum > 0)) return UNILM_ERR_INTERNAL;
  unilm_real_t inv = (unilm_real_t)1.0 / sum;

  for (size_t i = 0; i < m->vocab_size; i++) {
    unilm_real_t p = (unilm_real_t)exp((double)m->logp[i]) * inv;
    if (p < min_prob) p = min_prob;
    m->logp[i] = (unilm_real_t)log((double)p);
  }

  /* フロア後の再正規化 */
  sum = 0;
  for (size_t i = 0; i < m->vocab_size; i++) sum += (unilm_real_t)exp((double)m->logp[i]);
  if (!(sum > 0)) return UNILM_ERR_INTERNAL;
  inv = (unilm_real_t)1.0 / sum;
  for (size_t i = 0; i < m->vocab_size; i++) {
    unilm_real_t p = (unilm_real_t)exp((double)m->logp[i]) * inv;
    if (p < min_prob) p = min_prob;
    m->logp[i] = (unilm_real_t)log((double)p);
  }
  return UNILM_OK;
}

/*
 * 内部トライ（ダブル配列）を辞書順挿入で再構築する。
 *
 * 語彙の並び替えや部分更新後にトライを作り直したい場合に利用する。
 */
static const unilm_model_t *g_rebuild_trie_model = NULL;

static int cmp_piece_id_lex_bytes(const void *a, const void *b) {
  const uint32_t ia = *(const uint32_t *)a;
  const uint32_t ib = *(const uint32_t *)b;
  size_t la = 0, lb = 0;
  const uint8_t *pa = unilm_model_piece_bytes(g_rebuild_trie_model, ia, &la);
  const uint8_t *pb = unilm_model_piece_bytes(g_rebuild_trie_model, ib, &lb);
  size_t m = (la < lb) ? la : lb;
  int c = 0;
  if (m > 0) c = memcmp(pa, pb, m);
  if (c != 0) return c;
  if (la < lb) return -1;
  if (la > lb) return 1;
  /* tie-break by id for determinism */
  return (ia < ib) ? -1 : (ia > ib ? 1 : 0);
}

int unilm_model_rebuild_trie_sorted(unilm_model_t *m) {
  if (!m) return UNILM_ERR_BADARG;
  if (m->vocab_size == 0) {
    da_trie_clear(&m->trie);
    return UNILM_OK;
  }

  uint32_t *ids = (uint32_t *)malloc(m->vocab_size * sizeof(uint32_t));
  if (!ids) return UNILM_ERR_NOMEM;
  for (size_t i = 0; i < m->vocab_size; i++) ids[i] = (uint32_t)i;

  g_rebuild_trie_model = m;
  qsort(ids, m->vocab_size, sizeof(uint32_t), cmp_piece_id_lex_bytes);
  g_rebuild_trie_model = NULL;

  da_trie_clear(&m->trie);

  for (size_t idx = 0; idx < m->vocab_size; idx++) {
    uint32_t id = ids[idx];
    size_t blen = 0;
    const uint8_t *b = unilm_model_piece_bytes(m, id, &blen);
    if (!b || blen == 0) continue;
    if (da_trie_add_bytes(&m->trie, b, blen) != DA_OK) {
      free(ids);
      return UNILM_ERR_INTERNAL;
    }
    if (unilm_da_set_term_id(&m->trie, b, blen, id) != 0) {
      free(ids);
      return UNILM_ERR_INTERNAL;
    }
  }

  free(ids);
  return UNILM_OK;
}

/* ---------------- ワークスペース ---------------- */

static void unilm_workspace_zero(unilm_workspace_t *wk) {
  memset(wk, 0, sizeof(*wk));
}

#ifndef UNILM_NO_MALLOC

/* 動的アロケーションでワークスペースを初期化 */
int unilm_workspace_init_dynamic(unilm_workspace_t *wk,
                                size_t max_codepoints,
                                size_t vocab_cap,
                                size_t heap_cap) {
  if (!wk || max_codepoints == 0 || vocab_cap == 0) return UNILM_ERR_BADARG;
  unilm_workspace_zero(wk);

  size_t npos = max_codepoints + 1u;
  wk->cp_off = (uint32_t *)malloc(npos * sizeof(uint32_t));
  wk->alpha  = (unilm_real_t *)malloc(npos * sizeof(unilm_real_t));
  wk->beta   = (unilm_real_t *)malloc(npos * sizeof(unilm_real_t));
  wk->bp_prev  = (int32_t *)malloc(npos * sizeof(int32_t));
  wk->bp_piece = (int32_t *)malloc(npos * sizeof(int32_t));
  wk->keep = (uint8_t *)malloc(vocab_cap * sizeof(uint8_t));

  wk->heap_idx   = (heap_cap > 0) ? (uint32_t *)malloc(heap_cap * sizeof(uint32_t)) : NULL;
  wk->heap_score = (heap_cap > 0) ? (unilm_real_t *)malloc(heap_cap * sizeof(unilm_real_t)) : NULL;

  if (!wk->cp_off || !wk->alpha || !wk->beta || !wk->bp_prev || !wk->bp_piece || !wk->keep) {
    unilm_workspace_free(wk);
    return UNILM_ERR_NOMEM;
  }
  if (heap_cap > 0 && (!wk->heap_idx || !wk->heap_score)) {
    unilm_workspace_free(wk);
    return UNILM_ERR_NOMEM;
  }

  wk->cp_off_cap = npos;
  wk->dp_cap = npos;
  wk->bp_cap = npos;
  wk->keep_cap = vocab_cap;
  wk->heap_cap = heap_cap;
  wk->dynamic = 1;
  return UNILM_OK;
}

#else

int unilm_workspace_init_dynamic(unilm_workspace_t *wk,
                                size_t max_codepoints,
                                size_t vocab_cap,
                                size_t heap_cap) {
  (void)wk; (void)max_codepoints; (void)vocab_cap; (void)heap_cap;
  return UNILM_ERR_NOMEM;
}

#endif

/* 静的バッファでワークスペースを初期化 */
int unilm_workspace_init_static(unilm_workspace_t *wk,
                               uint32_t *cp_off, size_t cp_off_cap,
                               unilm_real_t *alpha, unilm_real_t *beta, size_t dp_cap,
                               int32_t *bp_prev, int32_t *bp_piece, size_t bp_cap,
                               uint8_t *keep, size_t keep_cap,
                               uint32_t *heap_idx, unilm_real_t *heap_score, size_t heap_cap) {
  if (!wk || !cp_off || !alpha || !beta || !keep) return UNILM_ERR_BADARG;
  if (cp_off_cap == 0 || dp_cap == 0 || keep_cap == 0) return UNILM_ERR_BADARG;
  unilm_workspace_zero(wk);

  wk->cp_off = cp_off;
  wk->cp_off_cap = cp_off_cap;
  wk->alpha = alpha;
  wk->beta = beta;
  wk->dp_cap = dp_cap;
  wk->bp_prev = bp_prev;
  wk->bp_piece = bp_piece;
  wk->bp_cap = bp_cap;
  wk->keep = keep;
  wk->keep_cap = keep_cap;
  wk->heap_idx = heap_idx;
  wk->heap_score = heap_score;
  wk->heap_cap = heap_cap;
  wk->dynamic = 0;
  return UNILM_OK;
}

/* ワークスペースを解放 */
void unilm_workspace_free(unilm_workspace_t *wk) {
  if (!wk) return;
#ifndef UNILM_NO_MALLOC
  if (wk->dynamic) {
    free(wk->cp_off);
    free(wk->alpha);
    free(wk->beta);
    free(wk->bp_prev);
    free(wk->bp_piece);
    free(wk->keep);
    free(wk->heap_idx);
    free(wk->heap_score);
  }
#endif
  unilm_workspace_zero(wk);
}

/* ---------------- EM コア ---------------- */

/*
 * コードポイント位置 i から始まるすべての一致ピースを列挙
 * BODY は __end_pos (size_t)、__pid (uint32_t) を受け取る
 */
#define UNILM_FOR_EACH_MATCH(m_, sent_, cp_off_, M_, i_, maxlen_, BODY) \
  do { \
    const da_index_t *__base = (m_)->trie.base; \
    const da_index_t *__check = (m_)->trie.check; \
    size_t __cap = (m_)->trie.capacity; \
    da_index_t __node = UNILM_DA_ROOT; \
    size_t __kmax = (i_) + (size_t)(((maxlen_) > 0) ? (maxlen_) : (int)(M_)); \
    if (__kmax > (M_)) __kmax = (M_); \
    for (size_t __k = (i_); __k < __kmax; __k++) { \
      size_t __b0 = (size_t)(cp_off_)[__k]; \
      size_t __b1 = (size_t)(cp_off_)[__k + 1u]; \
      for (size_t __b = __b0; __b < __b1; __b++) { \
        __node = unilm_da_next_raw(__base, __check, __cap, __node, (sent_)[__b]); \
        if (__node == 0) break; \
      } \
      if (__node == 0) break; \
      int32_t __tid = unilm_da_term_id(&(m_)->trie, __node); \
      if (__tid >= 0) { \
        size_t __end_pos = __k + 1u; \
        uint32_t __pid = (uint32_t)__tid; \
        BODY \
      } \
    } \
  } while (0)

/* 1 文に対する前向き-後ろ向きアルゴリズム */
static int unilm_forward_backward_sentence(const unilm_model_t *m,
                                          const uint8_t *sent, size_t sent_len,
                                          int max_piece_len_cp,
                                          unilm_workspace_t *wk,
                                          unilm_real_t *counts,
                                          unilm_real_t *out_logZ,
                                          unilm_real_t *out_tok_exp) {
  if (!m || !sent || !wk || !counts || !out_logZ) return UNILM_ERR_BADARG;
  if (!m->trie.base || !m->trie.check) return UNILM_ERR_BADARG;

  /* オフセット構築 */
  size_t M = 0;
  int rc = unilm_build_cp_offsets(sent, sent_len, wk->cp_off, wk->cp_off_cap, &M);
  if (rc != UNILM_OK) return rc;
  if (M + 1u > wk->dp_cap) return UNILM_ERR_RANGE;

  /* 前向き */
  for (size_t i = 0; i <= M; i++) wk->alpha[i] = (unilm_real_t)-INFINITY;
  wk->alpha[0] = (unilm_real_t)0;

  for (size_t i = 0; i < M; i++) {
    unilm_real_t ai = wk->alpha[i];
    if (!isfinite((double)ai)) continue;

    UNILM_FOR_EACH_MATCH(m, sent, wk->cp_off, M, i, max_piece_len_cp, {
      unilm_real_t cand = ai + m->logp[__pid];
      wk->alpha[__end_pos] = unilm_log_add(wk->alpha[__end_pos], cand);
    });
  }

  unilm_real_t logZ = wk->alpha[M];
  if (!isfinite((double)logZ)) return UNILM_ERR_NOCOVER;

  /* 後ろ向き */
  for (size_t i = 0; i <= M; i++) wk->beta[i] = (unilm_real_t)-INFINITY;
  wk->beta[M] = (unilm_real_t)0;

  for (size_t ii = M; ii-- > 0;) {
    size_t i = ii;
    unilm_real_t acc = (unilm_real_t)-INFINITY;

    UNILM_FOR_EACH_MATCH(m, sent, wk->cp_off, M, i, max_piece_len_cp, {
      unilm_real_t cand = m->logp[__pid] + wk->beta[__end_pos];
      acc = unilm_log_add(acc, cand);
    });

    wk->beta[i] = acc;
  }

  /* 期待カウント */
  unilm_real_t tok_exp = 0;
  for (size_t i = 0; i < M; i++) {
    unilm_real_t ai = wk->alpha[i];
    if (!isfinite((double)ai)) continue;

    UNILM_FOR_EACH_MATCH(m, sent, wk->cp_off, M, i, max_piece_len_cp, {
      unilm_real_t bj = wk->beta[__end_pos];
      unilm_real_t lp = m->logp[__pid];
      unilm_real_t log_use = ai + lp + bj - logZ;
      if (log_use > (unilm_real_t)-80) {
        unilm_real_t p = (unilm_real_t)exp((double)log_use);
        counts[__pid] += p;
        tok_exp += p;
      }
    });
  }

  *out_logZ = logZ;
  if (out_tok_exp) *out_tok_exp = tok_exp;
  return UNILM_OK;
}

/* E ステップ */
int unilm_em_e_step(const unilm_model_t *m,
                    const unilm_corpus_iter_t *it,
                    const unilm_train_config_t *cfg,
                    unilm_workspace_t *wk,
                    unilm_real_t *counts,
                    unilm_em_stats_t *out_stats) {
  if (!m || !it || !it->next || !cfg || !wk || !counts) return UNILM_ERR_BADARG;
  if (m->vocab_size == 0) return UNILM_ERR_BADARG;
  if (wk->cp_off_cap == 0 || wk->dp_cap == 0) return UNILM_ERR_BADARG;

  /* カウントをリセット */
  for (size_t i = 0; i < m->vocab_size; i++) counts[i] = (unilm_real_t)0;

  if (it->reset) it->reset(it->user);

  unilm_real_t sum_logZ = 0;
  unilm_real_t sum_tok = 0;
  size_t n_sent = 0;

  for (;;) {
    const uint8_t *s = NULL;
    size_t sl = 0;
    int r = it->next(it->user, &s, &sl);
    if (r == 0) break;
    if (r < 0) return UNILM_ERR_IO;
    if (!s || sl == 0) continue;

    unilm_real_t logZ = 0;
    unilm_real_t tok = 0;
    int rc = unilm_forward_backward_sentence(m, s, sl, cfg->max_piece_len_cp, wk, counts, &logZ, &tok);
    if (rc != UNILM_OK) return rc;
    sum_logZ += logZ;
    sum_tok += tok;
    n_sent++;
  }

  if (out_stats) {
    out_stats->loglik = sum_logZ;
    out_stats->n_sent = (unilm_real_t)n_sent;
    out_stats->n_tokens_exp = sum_tok;
  }
  return UNILM_OK;
}

/* M ステップ */
int unilm_em_m_step(unilm_model_t *m,
                    const unilm_train_config_t *cfg,
                    const unilm_real_t *counts) {
  if (!m || !cfg || !counts) return UNILM_ERR_BADARG;
  if (m->vocab_size == 0) return UNILM_ERR_BADARG;

  unilm_real_t smooth = cfg->smoothing;
  if (!(smooth >= 0)) smooth = (unilm_real_t)0;

  unilm_real_t total = 0;
  for (size_t i = 0; i < m->vocab_size; i++) {
    unilm_real_t c = counts[i] + smooth;
    if (!(c > 0)) c = (unilm_real_t)0;
    total += c;
  }
  if (!(total > 0)) return UNILM_ERR_INTERNAL;

  unilm_real_t minp = cfg->min_prob;
  if (!(minp > 0)) minp = (unilm_real_t)1e-12;

  for (size_t i = 0; i < m->vocab_size; i++) {
    unilm_real_t c = counts[i] + smooth;
    if (!(c > 0)) c = (unilm_real_t)0;
    unilm_real_t p = c / total;
    if (p < minp) p = minp;
    m->logp[i] = (unilm_real_t)log((double)p);
  }

  return unilm_model_normalize(m, minp);
}

/* ---------------- MDL スタイルの枝刈り ---------------- */

/* ピースが必須かどうかをチェック */
static inline int unilm_piece_is_mandatory(const unilm_model_t *m, uint32_t id) {
  const unilm_piece_t *p = &m->pieces[id];
  if (p->flags & UNILM_PIECE_MANDATORY) return 1;
  /* 単一コードポイントのピースは常に保持: カバレッジ保証に必要 */
  if (p->len_cp <= 1) return 1;
  return 0;
}

/*
 * ピースの文字フォールバックコストを計算: sum(-log p(char))
 * いずれかの文字が欠落していれば INFINITY を返す
 */
static unilm_real_t unilm_piece_char_cost(const unilm_model_t *m, uint32_t id) {
  size_t len = 0;
  const uint8_t *s = unilm_model_piece_bytes(m, id, &len);
  if (!s || len == 0) return (unilm_real_t)INFINITY;

  unilm_real_t cost = 0;
  for (size_t pos = 0; pos < len; ) {
    size_t next = unilm_utf8_next(s, len, pos);
    if (next <= pos) return (unilm_real_t)INFINITY;
    size_t clen = next - pos;
    int32_t cid = unilm_model_find_id(m, s + pos, clen);
    if (cid < 0) return (unilm_real_t)INFINITY;
    cost += -(m->logp[(uint32_t)cid]);
    pos = next;
  }
  return cost;
}

/* 上位 K スコア選択用の小さい最小ヒープ（インデックスを格納） */
static void heap_sift_up(uint32_t *idx, unilm_real_t *score, size_t i) {
  while (i > 0) {
    size_t p = (i - 1u) / 2u;
    if (score[p] <= score[i]) break;
    uint32_t ti = idx[p]; idx[p] = idx[i]; idx[i] = ti;
    unilm_real_t ts = score[p]; score[p] = score[i]; score[i] = ts;
    i = p;
  }
}

static void heap_sift_down(uint32_t *idx, unilm_real_t *score, size_t n, size_t i) {
  for (;;) {
    size_t l = 2u*i + 1u;
    size_t r = l + 1u;
    size_t s = i;
    if (l < n && score[l] < score[s]) s = l;
    if (r < n && score[r] < score[s]) s = r;
    if (s == i) break;
    uint32_t ti = idx[s]; idx[s] = idx[i]; idx[i] = ti;
    unilm_real_t ts = score[s]; score[s] = score[i]; score[i] = ts;
    i = s;
  }
}

/* MDL 枝刈り */
int unilm_prune_mdl(unilm_model_t *m,
                    const unilm_train_config_t *cfg,
                    unilm_workspace_t *wk,
                    const unilm_real_t *counts) {
  if (!m || !cfg || !wk || !counts) return UNILM_ERR_BADARG;
  if (m->vocab_size == 0) return UNILM_ERR_BADARG;

  size_t V = m->vocab_size;
  if (!wk->keep || wk->keep_cap < V) return UNILM_ERR_RANGE;

  memset(wk->keep, 0, V);

  /* 必須ピースをマーク */
  size_t mandatory = 0;
  for (uint32_t i = 0; i < (uint32_t)V; i++) {
    if (unilm_piece_is_mandatory(m, i)) {
      wk->keep[i] = 1;
      mandatory++;
    }
  }

  /* 枝刈りが要求されていなければそのまま保持 */
  int want_size_limit = (cfg->target_vocab_size > 0);
  if (!want_size_limit && !(cfg->mdl_lambda0 > 0 || cfg->mdl_lambda_len > 0)) {
    return (int)V;
  }

  /* 非必須ピースをスコアリング */
  unilm_real_t lambda0 = cfg->mdl_lambda0;
  unilm_real_t lambdalen = cfg->mdl_lambda_len;

  size_t K = 0;
  if (want_size_limit) {
    if (cfg->target_vocab_size <= mandatory) {
      K = 0;
    } else {
      K = cfg->target_vocab_size - mandatory;
    }
    if (K > wk->heap_cap) return UNILM_ERR_RANGE;
  }

  size_t heap_n = 0;

  for (uint32_t i = 0; i < (uint32_t)V; i++) {
    if (wk->keep[i]) continue;

    const unilm_piece_t *p = &m->pieces[i];
    unilm_real_t c = counts[i];
    if (!(c > 0)) c = (unilm_real_t)0;

    /* 文字分解に対する記述長削減 */
    unilm_real_t alt = unilm_piece_char_cost(m, i); /* >=0 */
    unilm_real_t self = -(m->logp[i]);
    if (!isfinite((double)alt) || !isfinite((double)self)) continue;

    unilm_real_t saved = (alt - self) * c;
    unilm_real_t cost = lambda0 + lambdalen * (unilm_real_t)p->len_cp;
    unilm_real_t score = saved - cost;

    if (!want_size_limit) {
      if (score > 0) wk->keep[i] = 1;
      continue;
    }

    /* サイズ制限: スコア上位 K を選択 */
    if (K == 0) continue;

    if (heap_n < K) {
      wk->heap_idx[heap_n] = i;
      wk->heap_score[heap_n] = score;
      heap_sift_up(wk->heap_idx, wk->heap_score, heap_n);
      heap_n++;
    } else {
      /* heap[0] が最小 */
      if (score > wk->heap_score[0]) {
        wk->heap_idx[0] = i;
        wk->heap_score[0] = score;
        heap_sift_down(wk->heap_idx, wk->heap_score, heap_n, 0);
      }
    }
  }

  if (want_size_limit) {
    for (size_t j = 0; j < heap_n; j++) {
      wk->keep[wk->heap_idx[j]] = 1;
    }
  }

  /* 語彙をインプレースで圧縮（文字列はプールに残り、オフセットは有効なまま） */
  size_t newV = 0;
  for (uint32_t i = 0; i < (uint32_t)V; i++) {
    if (wk->keep[i]) {
      if (newV != i) {
        m->pieces[newV] = m->pieces[i];
        m->logp[newV] = m->logp[i];
      }
      newV++;
    }
  }

  /* 新しい ID でトライを再構築 */
  if (da_trie_clear(&m->trie) != DA_OK) return UNILM_ERR_INTERNAL;
  for (uint32_t id = 0; id < (uint32_t)newV; id++) {
    size_t blen = 0;
    const uint8_t *b = unilm_model_piece_bytes(m, id, &blen);
    if (!b || blen == 0) return UNILM_ERR_INTERNAL;
    if (da_trie_add_bytes(&m->trie, b, blen) != DA_OK) return UNILM_ERR_FULL;
    if (unilm_da_set_term_id(&m->trie, b, blen, id) != UNILM_OK) return UNILM_ERR_INTERNAL;
  }

  m->vocab_size = newV;
  /* 確率の合計が 1 になることを保証（重複が移動した可能性あり） */
  (void)unilm_model_normalize(m, cfg->min_prob);
  return (int)newV;
}

/* ---------------- 完全学習ループ ---------------- */

int unilm_train_em_mdl(unilm_model_t *m,
                       const unilm_corpus_iter_t *it,
                       const unilm_train_config_t *cfg,
                       unilm_workspace_t *wk,
                       unilm_real_t *counts,
                       unilm_em_stats_t *out_last_stats) {
  if (!m || !it || !it->next || !cfg || !wk || !counts) return UNILM_ERR_BADARG;
  if (m->vocab_size == 0) return UNILM_ERR_BADARG;

  /* すべての logp がゼロ（一般的な初期状態）の場合、一様分布から開始 */
  int all_zero = 1;
  for (size_t i = 0; i < m->vocab_size; i++) {
    if (m->logp[i] != (unilm_real_t)0) { all_zero = 0; break; }
  }
  if (all_zero) {
    unilm_real_t lp = (unilm_real_t)(-log((double)m->vocab_size));
    for (size_t i = 0; i < m->vocab_size; i++) m->logp[i] = lp;
  }
  (void)unilm_model_normalize(m, cfg->min_prob);

  int iters = (cfg->num_iters > 0) ? cfg->num_iters : 1;
  unilm_em_stats_t st;
  memset(&st, 0, sizeof(st));

  for (int iter = 0; iter < iters; iter++) {
    int rc = unilm_em_e_step(m, it, cfg, wk, counts, &st);
    if (rc != UNILM_OK) return rc;

    rc = unilm_em_m_step(m, cfg, counts);
    if (rc != UNILM_OK) return rc;

    if (cfg->prune_each_iter) {
      rc = unilm_prune_mdl(m, cfg, wk, counts);
      if (rc < 0) return rc;
    }
  }

  if (out_last_stats) *out_last_stats = st;
  return UNILM_OK;
}

/* ---------------- 推論 (ビタビ) ---------------- */

int unilm_viterbi_tokenize(const unilm_model_t *m,
                           const uint8_t *sent, size_t sent_len,
                           int max_piece_len_cp,
                           unilm_workspace_t *wk,
                           uint32_t *out_ids, size_t out_cap,
                           size_t *out_n) {
  if (!m || !sent || !wk || !out_ids || !out_n) return UNILM_ERR_BADARG;
  if (!m->trie.base || !m->trie.check) return UNILM_ERR_BADARG;

  size_t M = 0;
  int rc = unilm_build_cp_offsets(sent, sent_len, wk->cp_off, wk->cp_off_cap, &M);
  if (rc != UNILM_OK) return rc;
  if (M + 1u > wk->dp_cap || M + 1u > wk->bp_cap) return UNILM_ERR_RANGE;

  for (size_t i = 0; i <= M; i++) {
    wk->alpha[i] = (unilm_real_t)-INFINITY;
    wk->bp_prev[i] = -1;
    wk->bp_piece[i] = -1;
  }
  wk->alpha[0] = (unilm_real_t)0;
  wk->bp_prev[0] = 0;

  for (size_t i = 0; i < M; i++) {
    unilm_real_t ai = wk->alpha[i];
    if (!isfinite((double)ai)) continue;

    UNILM_FOR_EACH_MATCH(m, sent, wk->cp_off, M, i, max_piece_len_cp, {
      unilm_real_t cand = ai + m->logp[__pid];
      if (cand > wk->alpha[__end_pos]) {
        wk->alpha[__end_pos] = cand;
        wk->bp_prev[__end_pos] = (int32_t)i;
        wk->bp_piece[__end_pos] = (int32_t)__pid;
      }
    });
  }

  if (!isfinite((double)wk->alpha[M]) || wk->bp_prev[M] < 0) return UNILM_ERR_NOCOVER;

  /* バックトレース */
  size_t n = 0;
  for (int32_t pos = (int32_t)M; pos > 0; ) {
    int32_t pid = wk->bp_piece[pos];
    int32_t prev = wk->bp_prev[pos];
    if (pid < 0 || prev < 0) return UNILM_ERR_INTERNAL;
    if (n >= out_cap) return UNILM_ERR_RANGE;
    out_ids[n++] = (uint32_t)pid;
    pos = prev;
  }

  /* 反転 */
  for (size_t i = 0; i < n/2; i++) {
    uint32_t t = out_ids[i];
    out_ids[i] = out_ids[n-1-i];
    out_ids[n-1-i] = t;
  }

  *out_n = n;
  return UNILM_OK;
}
