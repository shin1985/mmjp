/*
 * mmjp_train.c
 *
 * Wikipedia 等の大規模日本語コーパスから、
 *  - UniLM(MDL/EM)で単語（サブワード）を抽出し初期辞書を作成
 *  - 推論用に NPYCRF Lite モデル (CRF+LM) を組み立てて保存
 *
 * 目的:
 *  - MeCab/SentencePiece より「導入が軽い」「MCUで動く」分かち書き基盤
 *  - 品詞推定なし、未知語強い
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "../suffix_array/sa_utf8.h"
#include "../unilm_mdl/unilm_mdl.h"
#include "../npycrf_lite/npycrf_lite.h"
#include "mmjp_model.h"
#include "../mmjp_lossless.h"

/* npycrf_lite.h 側の NPYCRF_CC_* を使用（推論時 char_class() と完全一致） */
#define MMJP_CC_OTHER NPYCRF_CC_OTHER
#define MMJP_CC_SPACE NPYCRF_CC_SPACE
#define MMJP_CC_DIGIT NPYCRF_CC_DIGIT
#define MMJP_CC_ALPHA NPYCRF_CC_ALPHA
#define MMJP_CC_HIRAGANA NPYCRF_CC_HIRAGANA
#define MMJP_CC_KATAKANA NPYCRF_CC_KATAKANA
#define MMJP_CC_KANJI NPYCRF_CC_KANJI
#define MMJP_CC_FULLWIDTH NPYCRF_CC_FULLWIDTH
#define MMJP_CC_SYMBOL NPYCRF_CC_SYMBOL
#define MMJP_CC_BOS NPYCRF_CC_BOS
#define MMJP_CC_EOS NPYCRF_CC_EOS

/* =====================
 *  UTF-8 decode/encode (tool side)
 * ===================== */

static int utf8_decode1(const uint8_t *s, size_t len, size_t pos, uint32_t *out_cp, size_t *out_adv) {
  if (!s || pos >= len) return 0;
  uint8_t b0 = s[pos];
  if (b0 < 0x80u) {
    *out_cp = b0;
    *out_adv = 1;
    return 1;
  }
  if ((b0 & 0xE0u) == 0xC0u) {
    if (pos + 2 > len) return 0;
    uint8_t b1 = s[pos + 1];
    if ((b1 & 0xC0u) != 0x80u) return 0;
    *out_cp = ((uint32_t)(b0 & 0x1Fu) << 6) | (uint32_t)(b1 & 0x3Fu);
    *out_adv = 2;
    return 1;
  }
  if ((b0 & 0xF0u) == 0xE0u) {
    if (pos + 3 > len) return 0;
    uint8_t b1 = s[pos + 1], b2 = s[pos + 2];
    if ((b1 & 0xC0u) != 0x80u || (b2 & 0xC0u) != 0x80u) return 0;
    *out_cp = ((uint32_t)(b0 & 0x0Fu) << 12) | ((uint32_t)(b1 & 0x3Fu) << 6) | (uint32_t)(b2 & 0x3Fu);
    *out_adv = 3;
    return 1;
  }
  if ((b0 & 0xF8u) == 0xF0u) {
    if (pos + 4 > len) return 0;
    uint8_t b1 = s[pos + 1], b2 = s[pos + 2], b3 = s[pos + 3];
    if ((b1 & 0xC0u) != 0x80u || (b2 & 0xC0u) != 0x80u || (b3 & 0xC0u) != 0x80u) return 0;
    *out_cp = ((uint32_t)(b0 & 0x07u) << 18) | ((uint32_t)(b1 & 0x3Fu) << 12) | ((uint32_t)(b2 & 0x3Fu) << 6) | (uint32_t)(b3 & 0x3Fu);
    *out_adv = 4;
    return 1;
  }
  return 0;
}

static size_t utf8_encode1(uint32_t cp, uint8_t out[4]) {
  if (cp <= 0x7Fu) {
    out[0] = (uint8_t)cp;
    return 1;
  }
  if (cp <= 0x7FFu) {
    out[0] = (uint8_t)(0xC0u | ((cp >> 6) & 0x1Fu));
    out[1] = (uint8_t)(0x80u | (cp & 0x3Fu));
    return 2;
  }
  if (cp <= 0xFFFFu) {
    out[0] = (uint8_t)(0xE0u | ((cp >> 12) & 0x0Fu));
    out[1] = (uint8_t)(0x80u | ((cp >> 6) & 0x3Fu));
    out[2] = (uint8_t)(0x80u | (cp & 0x3Fu));
    return 3;
  }
  out[0] = (uint8_t)(0xF0u | ((cp >> 18) & 0x07u));
  out[1] = (uint8_t)(0x80u | ((cp >> 12) & 0x3Fu));
  out[2] = (uint8_t)(0x80u | ((cp >> 6) & 0x3Fu));
  out[3] = (uint8_t)(0x80u | (cp & 0x3Fu));
  return 4;
}

static size_t utf8_count_cp(const uint8_t *s, size_t len) {
  size_t n = 0;
  for (size_t pos = 0; pos < len; ) {
    uint32_t cp = 0;
    size_t adv = 0;
    if (!utf8_decode1(s, len, pos, &cp, &adv) || adv == 0) break;
    pos += adv;
    n++;
  }
  return n;
}

/* =====================
 *  uint32 set (open addressing)
 * ===================== */

typedef struct {
  uint32_t *k; /* 0 = empty */
  size_t cap;
  size_t size;
} u32set_t;

/*
 * codepoint -> count の簡易ハッシュ表
 *  - オフライン学習用なので、実装はシンプル優先
 */
typedef struct {
  uint32_t *cp;   /* 0 = empty */
  uint32_t *cnt;  /* count */
  size_t cap;
  size_t size;
} u32cnt_t;

static uint32_t u32_hash(uint32_t x) {
  /* simple mix */
  x ^= x >> 16;
  x *= 0x7feb352du;
  x ^= x >> 15;
  x *= 0x846ca68bu;
  x ^= x >> 16;
  return x;
}

static int u32set_init(u32set_t *s, size_t cap) {
  if (!s || cap < 16) return 0;
  s->k = (uint32_t *)calloc(cap, sizeof(uint32_t));
  if (!s->k) return 0;
  s->cap = cap;
  s->size = 0;
  return 1;
}

static void u32set_free(u32set_t *s) {
  if (!s) return;
  free(s->k);
  memset(s, 0, sizeof(*s));
}

static int u32cnt_init(u32cnt_t *m, size_t cap) {
  if (!m || cap < 16) return 0;
  m->cp = (uint32_t *)calloc(cap, sizeof(uint32_t));
  m->cnt = (uint32_t *)calloc(cap, sizeof(uint32_t));
  if (!m->cp || !m->cnt) {
    free(m->cp);
    free(m->cnt);
    memset(m, 0, sizeof(*m));
    return 0;
  }
  m->cap = cap;
  m->size = 0;
  return 1;
}

static void u32cnt_free(u32cnt_t *m) {
  if (!m) return;
  free(m->cp);
  free(m->cnt);
  memset(m, 0, sizeof(*m));
}

static int u32cnt_rehash(u32cnt_t *m, size_t newcap) {
  u32cnt_t n;
  if (!u32cnt_init(&n, newcap)) return 0;
  for (size_t i = 0; i < m->cap; i++) {
    uint32_t key = m->cp[i];
    if (key == 0) continue;
    uint32_t h = u32_hash(key);
    size_t mask = n.cap - 1;
    size_t idx = (size_t)h & mask;
    while (n.cp[idx] != 0) idx = (idx + 1) & mask;
    n.cp[idx] = key;
    n.cnt[idx] = m->cnt[i];
    n.size++;
  }
  free(m->cp);
  free(m->cnt);
  *m = n;
  return 1;
}

static int u32cnt_inc(u32cnt_t *m, uint32_t key) {
  if (!m || !m->cp || !m->cnt) return 0;
  if (key == 0) return 1;
  if ((m->size + 1) * 10 >= m->cap * 7) {
    size_t newcap = m->cap * 2;
    if (newcap < 16) newcap = 16;
    if ((newcap & (newcap - 1)) != 0) {
      size_t p = 16;
      while (p < newcap) p <<= 1;
      newcap = p;
    }
    if (!u32cnt_rehash(m, newcap)) return 0;
  }
  size_t mask = m->cap - 1;
  size_t idx = (size_t)u32_hash(key) & mask;
  while (m->cp[idx] != 0 && m->cp[idx] != key) idx = (idx + 1) & mask;
  if (m->cp[idx] == 0) {
    m->cp[idx] = key;
    m->cnt[idx] = 1;
    m->size++;
  } else {
    if (m->cnt[idx] < 0xFFFFFFFFu) m->cnt[idx]++;
  }
  return 1;
}

static int u32set_rehash(u32set_t *s, size_t newcap) {
  u32set_t n;
  if (!u32set_init(&n, newcap)) return 0;
  for (size_t i = 0; i < s->cap; i++) {
    uint32_t key = s->k[i];
    if (key == 0) continue;
    /* insert */
    uint32_t h = u32_hash(key);
    size_t mask = n.cap - 1;
    size_t idx = (size_t)h & mask;
    while (n.k[idx] != 0) idx = (idx + 1) & mask;
    n.k[idx] = key;
    n.size++;
  }
  free(s->k);
  *s = n;
  return 1;
}

static int u32set_insert(u32set_t *s, uint32_t key) {
  if (!s || !s->k) return 0;
  if (key == 0) return 1;
  if ((s->size + 1) * 10 >= s->cap * 7) {
    /* grow: cap must be power of 2 */
    size_t newcap = s->cap * 2;
    if (newcap < 16) newcap = 16;
    if ((newcap & (newcap - 1)) != 0) {
      /* ensure power of two */
      size_t p = 16;
      while (p < newcap) p <<= 1;
      newcap = p;
    }
    if (!u32set_rehash(s, newcap)) return 0;
  }
  size_t mask = s->cap - 1;
  size_t idx = (size_t)u32_hash(key) & mask;
  while (s->k[idx] != 0 && s->k[idx] != key) idx = (idx + 1) & mask;
  if (s->k[idx] == key) return 1;
  s->k[idx] = key;
  s->size++;
  return 1;
}

static int u32set_contains(const u32set_t *s, uint32_t key) {
  if (!s || !s->k || s->cap == 0) return 0;
  if (key == 0) return 0;
  size_t mask = s->cap - 1;
  size_t idx = (size_t)u32_hash(key) & mask;
  while (s->k[idx] != 0) {
    if (s->k[idx] == key) return 1;
    idx = (idx + 1) & mask;
  }
  return 0;
}

/* =====================
 *  file line iterator for UniLM
 * ===================== */

typedef struct {
  FILE *f;
  uint8_t *buf;
  size_t cap;
  size_t len;

  /* 入力行の上限（バイト）。0 = unlimited。超えた行は丸ごとスキップ */
  size_t max_line_bytes;

  /* 入力行の上限（コードポイント）。0 = unlimited。
   * skip_long_cp=1 のとき、cp_count > max_sentence_cp の行は丸ごとスキップ
   * （EM/DP ワークスペースに収まらない行で落ちないようにするため）
   */
  size_t max_sentence_cp;
  int    skip_long_cp;

  /* 低頻度文字をフォールバックに置換した行（UniLM学習/推論安定化用） */
  uint8_t *mapped;
  size_t mapped_cap;
  size_t mapped_len;

  /* 置換用 char-set */
  const u32set_t *keep_chars; /* NULLなら置換しない */
  uint32_t fallback_cp;       /* 例: '?' */

  /* stats (debug) */
  size_t last_cp;                 /* 直近の行のコードポイント数（mapping 後） */
  size_t stat_skipped_long_bytes; /* max_line_bytes でスキップした行数 */
  size_t stat_skipped_long_cp;    /* max_sentence_cp でスキップした行数 */

  /* lossless_eol: 各行末にメタLF（▃）を付与 */
  int append_eol;
} file_iter_t;

static int file_iter_reset(file_iter_t *it) {
  if (!it || !it->f) return 0;
  fseek(it->f, 0, SEEK_SET);
  return 1;
}

static int file_iter_readline(file_iter_t *it) {
  if (!it || !it->f) return -1;
  it->len = 0;
  it->last_cp = 0;
  int c;
  while ((c = fgetc(it->f)) != EOF) {
    if (c == '\n') break;
    if (it->max_line_bytes > 0 && it->len >= it->max_line_bytes) {
      /* discard until end of line */
      while (c != EOF && c != '\n') c = fgetc(it->f);
      it->len = 0;
      it->stat_skipped_long_bytes++;
      /* スキップした「1行」として扱い、呼び出し側で空行として捨てる */
      it->mapped_len = 0;
      return 1;
    }
    if (it->len + 1 > it->cap) {
      size_t newcap = it->cap ? it->cap * 2 : 256;
      uint8_t *nb = (uint8_t *)realloc(it->buf, newcap);
      if (!nb) return -2;
      it->buf = nb;
      it->cap = newcap;
    }
    it->buf[it->len++] = (uint8_t)c;
  }
  if (c == EOF && it->len == 0) return 0; /* end */

  /* trim CR */
  while (it->len > 0 && (it->buf[it->len - 1] == '\r' || it->buf[it->len - 1] == ' ' || it->buf[it->len - 1] == '\t')) {
    it->len--;
  }

  /* append meta-LF (▃ U+2583 = 0xE2 0x96 0x83) if lossless_eol is enabled */
  if (it->append_eol && it->len > 0) {
    if (it->len + 3 > it->cap) {
      size_t newcap = it->cap ? it->cap * 2 : 256;
      while (newcap < it->len + 3) newcap *= 2;
      uint8_t *nb = (uint8_t *)realloc(it->buf, newcap);
      if (!nb) return -2;
      it->buf = nb;
      it->cap = newcap;
    }
    it->buf[it->len++] = 0xE2;
    it->buf[it->len++] = 0x96;
    it->buf[it->len++] = 0x83;
  }

  /* optional: map rare chars -> fallback */
  it->mapped_len = 0;
  if (it->keep_chars) {
    /* fallback は 1 codepoint (通常 '?' ) */
    uint8_t fb[4];
    size_t fb_len = utf8_encode1(it->fallback_cp, fb);

    /* '?' の場合、出力は入力より短くなることが多いので len+1 で十分 */
    size_t need = it->len + 1;
    if (it->mapped_cap < need) {
      size_t nc = it->mapped_cap ? it->mapped_cap * 2 : 256;
      while (nc < need) nc *= 2;
      uint8_t *nb = (uint8_t *)realloc(it->mapped, nc);
      if (!nb) return -3;
      it->mapped = nb;
      it->mapped_cap = nc;
    }

    const uint8_t *s = it->buf;
    size_t pos = 0;
    size_t outp = 0;
    size_t cp_count = 0;
    while (pos < it->len) {
      cp_count++;
      if (it->skip_long_cp && it->max_sentence_cp > 0 && cp_count > it->max_sentence_cp) {
        /* 行が長すぎる（コードポイント数）。ワークスペース外なのでスキップ */
        it->len = 0;
        it->mapped_len = 0;
        it->last_cp = cp_count;
        it->stat_skipped_long_cp++;
        return 1;
      }
      uint32_t cp = 0;
      size_t adv = 0;
      int ok = utf8_decode1(s, it->len, pos, &cp, &adv);
      if (!ok || adv == 0) {
        /* 不正な UTF-8 バイト列は常にフォールバックに落とす（raw byte を保持しない） */
        if (outp + fb_len > it->mapped_cap) {
          size_t nc = it->mapped_cap ? it->mapped_cap * 2 : 256;
          while (nc < outp + fb_len + 1) nc *= 2;
          uint8_t *nb = (uint8_t *)realloc(it->mapped, nc);
          if (!nb) return -5;
          it->mapped = nb;
          it->mapped_cap = nc;
        }
        memcpy(&it->mapped[outp], fb, fb_len);
        outp += fb_len;
        pos += 1;
        continue;
      }

      /* invalid Unicode scalar (surrogates / >U+10FFFF) -> fallback */
      if (cp > 0x10FFFFu || (cp >= 0xD800u && cp <= 0xDFFFu)) {
        if (outp + fb_len > it->mapped_cap) {
          size_t nc = it->mapped_cap ? it->mapped_cap * 2 : 256;
          while (nc < outp + fb_len + 1) nc *= 2;
          uint8_t *nb = (uint8_t *)realloc(it->mapped, nc);
          if (!nb) return -5;
          it->mapped = nb;
          it->mapped_cap = nc;
        }
        memcpy(&it->mapped[outp], fb, fb_len);
        outp += fb_len;
        pos += adv;
        continue;
      }

      if (u32set_contains(it->keep_chars, cp)) {
        /* keep char: normalize to canonical UTF-8 bytes (avoid overlong sequences) */
        uint8_t enc[4];
        size_t enc_len = utf8_encode1(cp, enc);
        if (outp + enc_len > it->mapped_cap) {
          size_t nc = it->mapped_cap ? it->mapped_cap * 2 : 256;
          while (nc < outp + enc_len + 1) nc *= 2;
          uint8_t *nb = (uint8_t *)realloc(it->mapped, nc);
          if (!nb) return -4;
          it->mapped = nb;
          it->mapped_cap = nc;
        }
        memcpy(&it->mapped[outp], enc, enc_len);
        outp += enc_len;
      } else {
        if (outp + fb_len > it->mapped_cap) {
          size_t nc = it->mapped_cap ? it->mapped_cap * 2 : 256;
          while (nc < outp + fb_len + 1) nc *= 2;
          uint8_t *nb = (uint8_t *)realloc(it->mapped, nc);
          if (!nb) return -5;
          it->mapped = nb;
          it->mapped_cap = nc;
        }
        memcpy(&it->mapped[outp], fb, fb_len);
        outp += fb_len;
      }
      pos += adv;
    }
    it->last_cp = cp_count;
    it->mapped_len = outp;
    it->mapped[outp] = 0;
  }
  return 1;
}

static int file_corpus_next(void *user, const uint8_t **out, size_t *out_len) {
  file_iter_t *it = (file_iter_t *)user;
  if (!it || !out || !out_len) return -1;
  for (;;) {
    int r = file_iter_readline(it);
    if (r < 0) return r;
    if (r == 0) return 0;
    if (it->len == 0) continue; /* skip empty */
    if (it->keep_chars && it->mapped_len > 0) {
      *out = it->mapped;
      *out_len = it->mapped_len;
    } else {
      *out = it->buf;
      *out_len = it->len;
    }
    return 1;
  }
}

static void file_corpus_reset(void *user) {
  file_iter_reset((file_iter_t *)user);
}

/* =====================
 *  coverage / debug helpers
 * ===================== */

static void mmjp_dump_nocover_details(const unilm_model_t *um, const uint8_t *s, size_t sl) {
  if (!um || !s || sl == 0) return;
  fprintf(stderr, "[nocover] dump: bytes=%zu\n", sl);

  /* まずはそのまま出す（UTF-8なら読める）。巨大行は切る */
  size_t preview = sl;
  if (preview > 400) preview = 400;
  fprintf(stderr, "[nocover] preview(<=400B): ");
  fwrite(s, 1, preview, stderr);
  if (preview < sl) fprintf(stderr, "...");
  fputc('\n', stderr);

  /* どの single が欠けているか探す */
  size_t pos = 0;
  size_t cp_i = 0;
  int shown = 0;
  while (pos < sl) {
    uint32_t cp = 0;
    size_t adv = 0;
    int ok = utf8_decode1(s, sl, pos, &cp, &adv);
    if (!ok || adv == 0) {
      /* 念のため */
      cp = (uint32_t)s[pos];
      adv = 1;
    }
    int32_t id = unilm_model_find_id(um, s + pos, adv);
    if (id < 0) {
      fprintf(stderr, "[nocover] missing single: cp_index=%zu byte_pos=%zu cp=U+%04X bytes=", cp_i, pos, cp);
      for (size_t k = 0; k < adv; k++) fprintf(stderr, "%02X", (unsigned)(s[pos + k]));
      fprintf(stderr, "\n");
      if (++shown >= 10) break;
    }
    pos += adv;
    cp_i++;
  }

  if (shown == 0) {
    fprintf(stderr, "[nocover] note: all single-char pieces for this sentence seem present, but tokenize still fails.\n");
    fprintf(stderr, "          this usually indicates trie corruption or a mismatch between mapping and vocab bytes.\n");
  }
}

static int mmjp_locate_first_nocover(const unilm_model_t *um,
                                    file_iter_t *fit,
                                    unilm_workspace_t *wk,
                                    int max_piece_len_cp,
                                    size_t out_cap,
                                    size_t limit_sent) {
  if (!um || !fit || !wk || out_cap == 0) return -1;
  uint32_t *ids = (uint32_t *)malloc(out_cap * sizeof(uint32_t));
  if (!ids) return -2;

  file_iter_reset(fit);

  size_t n_sent = 0;
  for (;;) {
    const uint8_t *s = NULL;
    size_t sl = 0;
    int r = file_corpus_next(fit, &s, &sl);
    if (r == 0) break;
    if (r < 0) { free(ids); return -3; }
    if (!s || sl == 0) continue;
    n_sent++;
    if (limit_sent > 0 && n_sent > limit_sent) break;

    size_t out_n = 0;
    int rc = unilm_viterbi_tokenize(um, s, sl, max_piece_len_cp, wk, ids, out_cap, &out_n);
    if (rc == UNILM_ERR_NOCOVER) {
      fprintf(stderr, "[nocover] first failing sentence=%zu (len=%zu bytes)\n", n_sent, sl);
      mmjp_dump_nocover_details(um, s, sl);
      free(ids);
      return 1;
    }
    if (rc == UNILM_ERR_RANGE) {
      /* 文が長すぎる場合。EM側は RANGE で落ちるはずだが、念のため案内 */
      fprintf(stderr, "[warn] viterbi RANGE at sentence=%zu (len=%zu bytes). consider --max_sentence_cp\n", n_sent, sl);
      continue;
    }
    if (rc != UNILM_OK) {
      fprintf(stderr, "[warn] viterbi rc=%d at sentence=%zu\n", rc, n_sent);
    }
  }

  free(ids);
  return 0;
}

/* =====================
 * candidate heap
 * ===================== */

typedef struct {
  uint32_t count;
  uint16_t len_bytes;
  uint16_t len_cp;
  char *s; /* null terminated */
} cand_t;

typedef struct {
  cand_t *a;
  size_t n;
  size_t cap;
} cand_heap_t;

static void cand_free(cand_t *c) {
  if (!c) return;
  free(c->s);
  c->s = NULL;
}

static void heap_swap(cand_t *x, cand_t *y) {
  cand_t t = *x;
  *x = *y;
  *y = t;
}

static void heap_sift_up(cand_heap_t *h, size_t i) {
  while (i > 0) {
    size_t p = (i - 1) / 2;
    if (h->a[p].count <= h->a[i].count) break;
    heap_swap(&h->a[p], &h->a[i]);
    i = p;
  }
}

static void heap_sift_down(cand_heap_t *h, size_t i) {
  for (;;) {
    size_t l = i * 2 + 1, r = l + 1;
    size_t m = i;
    if (l < h->n && h->a[l].count < h->a[m].count) m = l;
    if (r < h->n && h->a[r].count < h->a[m].count) m = r;
    if (m == i) break;
    heap_swap(&h->a[m], &h->a[i]);
    i = m;
  }
}

static int heap_init(cand_heap_t *h, size_t cap) {
  if (!h) return 0;
  h->a = (cand_t *)calloc(cap, sizeof(cand_t));
  if (!h->a) return 0;
  h->n = 0;
  h->cap = cap;
  return 1;
}

static void heap_free(cand_heap_t *h) {
  if (!h) return;
  for (size_t i = 0; i < h->n; i++) cand_free(&h->a[i]);
  free(h->a);
  memset(h, 0, sizeof(*h));
}

static int heap_push_topk(cand_heap_t *h, uint32_t count, const char *s, uint16_t len_bytes, uint16_t len_cp) {
  if (!h || !h->a || h->cap == 0) return 0;
  if (h->n < h->cap) {
    cand_t *c = &h->a[h->n++];
    c->count = count;
    c->len_bytes = len_bytes;
    c->len_cp = len_cp;
    c->s = (char *)malloc((size_t)len_bytes + 1u);
    if (!c->s) return 0;
    memcpy(c->s, s, len_bytes);
    c->s[len_bytes] = '\0';
    heap_sift_up(h, h->n - 1);
    return 1;
  }
  /* full */
  if (count <= h->a[0].count) return 1;
  /* replace root */
  cand_free(&h->a[0]);
  h->a[0].count = count;
  h->a[0].len_bytes = len_bytes;
  h->a[0].len_cp = len_cp;
  h->a[0].s = (char *)malloc((size_t)len_bytes + 1u);
  if (!h->a[0].s) return 0;
  memcpy(h->a[0].s, s, len_bytes);
  h->a[0].s[len_bytes] = '\0';
  heap_sift_down(h, 0);
  return 1;
}

/* =====================
 * misc helpers
 * ===================== */

static int is_bad_byte(uint8_t b) {
  return (b == 0u || b == '\n' || b == '\r' || b == '\t');
}

static int is_good_piece_bytes(const char *s, size_t len) {
  for (size_t i = 0; i < len; i++) {
    uint8_t b = (uint8_t)s[i];
    if (is_bad_byte(b)) return 0;
    if (b == ' ') return 0;
  }
  return 1;
}

static int cmp_cand_desc(const void *a, const void *b) {
  const cand_t *x = (const cand_t *)a;
  const cand_t *y = (const cand_t *)b;
  if (x->count < y->count) return 1;
  if (x->count > y->count) return -1;
  return (int)x->len_cp - (int)y->len_cp;
}

static int cmp_double_desc(const void *a, const void *b) {
  double x = *(const double *)a;
  double y = *(const double *)b;
  return (x < y) ? 1 : (x > y) ? -1 : 0;
}

static int16_t q88_from_double(double v) {
  /* clamp */
  double x = v * 256.0;
  if (x > 32767.0) x = 32767.0;
  if (x < -32768.0) x = -32768.0;
  return (int16_t)lrint(x);
}

/* =====================
 * CRF preset (ja_basic)
 * ===================== */

typedef struct {
  uint32_t *k;
  int16_t *w;
  uint32_t n;
} crf_table_t;

static void crf_table_free(crf_table_t *t) {
  if (!t) return;
  free(t->k);
  free(t->w);
  memset(t, 0, sizeof(*t));
}

/* helper for sorting feature table build */
typedef struct {
  uint32_t k;
  int16_t w;
} crf_kv_t;

static int cmp_crf_kv_key(const void *pa, const void *pb) {
  const crf_kv_t *x = (const crf_kv_t *)pa;
  const crf_kv_t *y = (const crf_kv_t *)pb;
  if (x->k < y->k) return -1;
  if (x->k > y->k) return 1;
  return 0;
}

static int crf_table_build_ja_basic(crf_table_t *out) {
  if (!out) return 0;
  memset(out, 0, sizeof(*out));

  /* Character class sets.
   * - cur: actual char classes (exclude BOS/EOS)
   * - prev: BOS + actual
   * - next: actual + EOS
   */
  const uint8_t cur_cls[] = {
      MMJP_CC_OTHER, MMJP_CC_SPACE, MMJP_CC_DIGIT, MMJP_CC_ALPHA,
      MMJP_CC_HIRAGANA, MMJP_CC_KATAKANA, MMJP_CC_KANJI, MMJP_CC_FULLWIDTH, MMJP_CC_SYMBOL,
  };
  const uint8_t prev_cls[] = {
      MMJP_CC_BOS,
      MMJP_CC_OTHER, MMJP_CC_SPACE, MMJP_CC_DIGIT, MMJP_CC_ALPHA,
      MMJP_CC_HIRAGANA, MMJP_CC_KATAKANA, MMJP_CC_KANJI, MMJP_CC_FULLWIDTH, MMJP_CC_SYMBOL,
  };
  const uint8_t next_cls[] = {
      MMJP_CC_OTHER, MMJP_CC_SPACE, MMJP_CC_DIGIT, MMJP_CC_ALPHA,
      MMJP_CC_HIRAGANA, MMJP_CC_KATAKANA, MMJP_CC_KANJI, MMJP_CC_FULLWIDTH, MMJP_CC_SYMBOL,
      MMJP_CC_EOS,
  };

  const size_t n_cur = sizeof(cur_cls) / sizeof(cur_cls[0]);
  const size_t n_prev = sizeof(prev_cls) / sizeof(prev_cls[0]);
  const size_t n_next = sizeof(next_cls) / sizeof(next_cls[0]);

  /* total features:
   *  tid0 (cur)  : 2 * n_cur
   *  tid1 (prev) : 2 * n_prev
   *  tid2 (next) : 2 * n_next
   *  tid3 (prev-cur pair): 2 * (n_prev * n_cur)
   *  tid4 (cur-next pair): 2 * (n_cur * n_next)
   */
  const size_t n = 2u * (n_cur + n_prev + n_next) + 2u * (n_prev * n_cur) + 2u * (n_cur * n_next);

  crf_kv_t *a = (crf_kv_t *)malloc(n * sizeof(crf_kv_t));
  if (!a) return 0;

  /* Defaults: keep previous "tiny heuristic" behavior by assigning non-zero
   * weights only to a handful of unary features. The rest are 0 and can be
   * learned via --crf_supervised (SGD/L-BFGS) or overridden by --crf_config.
   */
  struct init_feat {
    uint8_t tid, label, v1, v2;
    double w;
  } init[] = {
      /* prev_class -> start */
      {1, 1, MMJP_CC_BOS, 0, 2.0},
      {1, 1, MMJP_CC_SPACE, 0, 1.5},
      {1, 1, MMJP_CC_SYMBOL, 0, 1.2},
      {1, 1, MMJP_CC_FULLWIDTH, 0, 1.2},

      /* cur_class -> start */
      {0, 1, MMJP_CC_SPACE, 0, 1.5},
      {0, 1, MMJP_CC_SYMBOL, 0, 1.5},
      {0, 1, MMJP_CC_FULLWIDTH, 0, 1.5},

      /* cur_class -> internal */
      {0, 0, MMJP_CC_SPACE, 0, -2.0},
      {0, 0, MMJP_CC_SYMBOL, 0, -2.0},
      {0, 0, MMJP_CC_FULLWIDTH, 0, -2.0},

      {0, 0, MMJP_CC_KANJI, 0, 0.4},
      {0, 0, MMJP_CC_KATAKANA, 0, 0.4},
      {0, 0, MMJP_CC_ALPHA, 0, 0.2},
      {0, 0, MMJP_CC_DIGIT, 0, 0.2},
      {0, 0, MMJP_CC_HIRAGANA, 0, 0.1},
  };
  const size_t n_init = sizeof(init) / sizeof(init[0]);

  size_t idx = 0;

  for (uint8_t label = 0; label <= 1; label++) {
    /* tid0: cur */
    for (size_t i = 0; i < n_cur; i++) {
      uint32_t key = npycrf_feat_key(0, label, cur_cls[i], 0);
      a[idx].k = key;
      a[idx].w = 0;
      idx++;
    }
    /* tid1: prev */
    for (size_t i = 0; i < n_prev; i++) {
      uint32_t key = npycrf_feat_key(1, label, prev_cls[i], 0);
      a[idx].k = key;
      a[idx].w = 0;
      idx++;
    }
    /* tid2: next */
    for (size_t i = 0; i < n_next; i++) {
      uint32_t key = npycrf_feat_key(2, label, next_cls[i], 0);
      a[idx].k = key;
      a[idx].w = 0;
      idx++;
    }
    /* tid3: prev-cur pair */
    for (size_t i = 0; i < n_prev; i++) {
      for (size_t j = 0; j < n_cur; j++) {
        uint32_t key = npycrf_feat_key(3, label, prev_cls[i], cur_cls[j]);
        a[idx].k = key;
        a[idx].w = 0;
        idx++;
      }
    }
    /* tid4: cur-next pair */
    for (size_t i = 0; i < n_cur; i++) {
      for (size_t j = 0; j < n_next; j++) {
        uint32_t key = npycrf_feat_key(4, label, cur_cls[i], next_cls[j]);
        a[idx].k = key;
        a[idx].w = 0;
        idx++;
      }
    }
  }

  if (idx != n) {
    free(a);
    return 0;
  }

  /* apply initial non-zero weights */
  for (size_t k = 0; k < n_init; k++) {
    uint32_t key = npycrf_feat_key(init[k].tid, init[k].label, init[k].v1, init[k].v2);
    for (size_t i = 0; i < n; i++) {
      if (a[i].k == key) {
        a[i].w = q88_from_double(init[k].w);
        break;
      }
    }
  }
  /* sort by key */
  qsort(a, n, sizeof(crf_kv_t), cmp_crf_kv_key);

  out->k = (uint32_t *)malloc(n * sizeof(uint32_t));
  out->w = (int16_t *)malloc(n * sizeof(int16_t));
  if (!out->k || !out->w) {
    free(a);
    crf_table_free(out);
    return 0;
  }
  out->n = (uint32_t)n;

  for (size_t i = 0; i < n; i++) {
    out->k[i] = a[i].k;
    out->w[i] = a[i].w;
  }

  free(a);
  return 1;
}

/* =====================
 * CRF weights: config + supervised training
 * ===================== */

static double q88_to_double(int16_t q) { return (double)q / 256.0; }

static int crf_table_find_idx(const crf_table_t *t, uint32_t key) {
  if (!t || !t->k || t->n == 0) return -1;
  size_t lo = 0, hi = (size_t)t->n;
  while (lo < hi) {
    size_t mid = lo + (hi - lo) / 2;
    uint32_t mk = t->k[mid];
    if (mk == key) return (int)mid;
    if (mk < key) lo = mid + 1;
    else hi = mid;
  }
  return -1;
}

static uint8_t mmjp_char_class_by_cp(uint32_t cp) {
  if (cp == 0) return MMJP_CC_OTHER;
  if (cp == 0x20u || cp == 0x09u || cp == 0x0Au || cp == 0x0Du) return MMJP_CC_SPACE;
  if (cp >= '0' && cp <= '9') return MMJP_CC_DIGIT;
  if ((cp >= 'A' && cp <= 'Z') || (cp >= 'a' && cp <= 'z')) return MMJP_CC_ALPHA;

  /* Hiragana */
  if (cp >= 0x3040u && cp <= 0x309Fu) return MMJP_CC_HIRAGANA;
  /* Katakana */
  if ((cp >= 0x30A0u && cp <= 0x30FFu) || (cp >= 0x31F0u && cp <= 0x31FFu)) return MMJP_CC_KATAKANA;
  /* Kanji (CJK Unified Ideographs) */
  if ((cp >= 0x4E00u && cp <= 0x9FFFu) || (cp >= 0x3400u && cp <= 0x4DBFu)) return MMJP_CC_KANJI;

  /* fullwidth ASCII variants */
  if (cp >= 0xFF01u && cp <= 0xFF60u) {
    if (cp >= 0xFF10u && cp <= 0xFF19u) return MMJP_CC_DIGIT;
    if ((cp >= 0xFF21u && cp <= 0xFF3Au) || (cp >= 0xFF41u && cp <= 0xFF5Au)) return MMJP_CC_ALPHA;
    return MMJP_CC_FULLWIDTH;
  }

  /* symbols/punct */
  if ((cp >= 0x2000u && cp <= 0x206Fu) || (cp >= 0x3000u && cp <= 0x303Fu) || (cp >= 0xFF61u && cp <= 0xFF65u)) {
    return MMJP_CC_SYMBOL;
  }

  return MMJP_CC_OTHER;
}

static int crf_apply_config_file(const char *path,
                                 double *trans00, double *trans01, double *trans10, double *trans11,
                                 double *bos_to1,
                                 const crf_table_t *tbl,
                                 double *feat_w /* length tbl->n */) {
  if (!path) return 1;
  FILE *f = fopen(path, "rb");
  if (!f) {
    fprintf(stderr, "[mmjp_train] CRF config: cannot open %s\n", path);
    return 0;
  }
  char line[512];
  int ok = 1;
  while (fgets(line, sizeof(line), f)) {
    /* trim */
    char *s = line;
    while (*s == ' ' || *s == '\t') s++;
    if (*s == 0 || *s == '\n' || *s == '\r') continue;
    if (*s == '#' || *s == ';') continue;

    /* strip trailing */
    size_t sl = strlen(s);
    while (sl > 0 && (s[sl - 1] == '\n' || s[sl - 1] == '\r')) s[--sl] = 0;

    double v = 0.0;
    if (sscanf(s, "trans00 = %lf", &v) == 1 || sscanf(s, "trans00=%lf", &v) == 1) {
      if (trans00) *trans00 = v;
      continue;
    }
    if (sscanf(s, "trans01 = %lf", &v) == 1 || sscanf(s, "trans01=%lf", &v) == 1) {
      if (trans01) *trans01 = v;
      continue;
    }
    if (sscanf(s, "trans10 = %lf", &v) == 1 || sscanf(s, "trans10=%lf", &v) == 1) {
      if (trans10) *trans10 = v;
      continue;
    }
    if (sscanf(s, "trans11 = %lf", &v) == 1 || sscanf(s, "trans11=%lf", &v) == 1) {
      if (trans11) *trans11 = v;
      continue;
    }
    if (sscanf(s, "bos_to1 = %lf", &v) == 1 || sscanf(s, "bos_to1=%lf", &v) == 1) {
      if (bos_to1) *bos_to1 = v;
      continue;
    }

    int tid = 0, label = 0, v1 = 0, v2 = 0;
    double w = 0.0;
    if (sscanf(s, "feat %d %d %d %d = %lf", &tid, &label, &v1, &v2, &w) == 5 ||
        sscanf(s, "feat %d %d %d %d %lf", &tid, &label, &v1, &v2, &w) == 5) {
      uint32_t key = npycrf_feat_key((uint8_t)tid, (uint8_t)label, (uint8_t)v1, (uint8_t)v2);
      int idx = crf_table_find_idx(tbl, key);
      if (idx >= 0 && feat_w) {
        feat_w[(size_t)idx] = w;
      } else {
        fprintf(stderr, "[mmjp_train] CRF config: unknown feature (tid=%d label=%d v1=%d v2=%d)\n",
                tid, label, v1, v2);
      }
      continue;
    }

    fprintf(stderr, "[mmjp_train] CRF config: ignored line: %s\n", s);
  }
  fclose(f);
  return ok;
}

typedef struct {
  uint8_t *cls; /* char class per codepoint */
  uint8_t *y;   /* label: 1 = start, 0 = internal */
  uint16_t n;   /* length */
} crf_sent_t;

typedef struct {
  crf_sent_t *s;
  size_t n;
  size_t cap;
  size_t total_pos;
} crf_dataset_t;

static void crf_dataset_free(crf_dataset_t *ds) {
  if (!ds) return;
  for (size_t i = 0; i < ds->n; i++) {
    free(ds->s[i].cls);
    free(ds->s[i].y);
  }
  free(ds->s);
  memset(ds, 0, sizeof(*ds));
}

static int crf_dataset_push(crf_dataset_t *ds, uint8_t *cls, uint8_t *y, uint16_t n) {
  if (!ds || !cls || !y || n == 0) return 0;
  if (ds->n + 1 > ds->cap) {
    size_t nc = ds->cap ? ds->cap * 2 : 256;
    crf_sent_t *ns = (crf_sent_t *)realloc(ds->s, nc * sizeof(crf_sent_t));
    if (!ns) return 0;
    ds->s = ns;
    ds->cap = nc;
  }
  ds->s[ds->n].cls = cls;
  ds->s[ds->n].y = y;
  ds->s[ds->n].n = n;
  ds->n++;
  ds->total_pos += n;
  return 1;
}

/* segmented line format: tokens separated by spaces/tabs.
 * Example: "東京 都 に 住んで い ます" (space-separated).
 */
static int crf_parse_segmented_line(const uint8_t *line, size_t len,
                                    uint8_t **out_cls, uint8_t **out_y, uint16_t *out_n,
                                    size_t max_sentence_cp) {
  if (!line || len == 0 || !out_cls || !out_y || !out_n) return 0;
  *out_cls = NULL;
  *out_y = NULL;
  *out_n = 0;

  size_t cap = 0;
  size_t n = 0;
  uint8_t *cls = NULL;
  uint8_t *y = NULL;
  size_t pos = 0;
  int at_token_start = 1;

  while (pos < len) {
    /* skip whitespace */
    while (pos < len && (line[pos] == ' ' || line[pos] == '\t')) {
      pos++;
      at_token_start = 1;
    }
    if (pos >= len) break;
    /* token */
    while (pos < len && line[pos] != ' ' && line[pos] != '\t') {
      uint32_t cp = 0;
      size_t adv = 0;
      if (!utf8_decode1(line, len, pos, &cp, &adv) || adv == 0) {
        free(cls);
        free(y);
        return 0;
      }
      pos += adv;
      if (max_sentence_cp > 0 && n + 1 > max_sentence_cp) {
        /* skip too long sentence */
        free(cls);
        free(y);
        return 0;
      }
      if (n + 1 > cap) {
        size_t nc = cap ? cap * 2 : 256;
        uint8_t *ncls = (uint8_t *)realloc(cls, nc);
        if (!ncls) {
          free(cls);
          free(y);
          return 0;
        }
        cls = ncls;
        uint8_t *ny = (uint8_t *)realloc(y, nc);
        if (!ny) {
          free(cls);
          free(y);
          return 0;
        }
        y = ny;
        cap = nc;
      }
      cls[n] = mmjp_char_class_by_cp(cp);
      y[n] = (uint8_t)(at_token_start ? 1 : 0);
      at_token_start = 0;
      n++;
    }
    at_token_start = 1;
  }

  if (n == 0) {
    free(cls);
    free(y);
    return 0;
  }
  /* enforce y[0]=1 */
  y[0] = 1;

  *out_cls = cls;
  *out_y = y;
  *out_n = (uint16_t)n;
  return 1;
}

static int crf_dataset_load(const char *path, size_t max_line_bytes, size_t max_sentence_cp, crf_dataset_t *out) {
  if (!path || !out) return 0;
  memset(out, 0, sizeof(*out));
  FILE *f = fopen(path, "rb");
  if (!f) {
    fprintf(stderr, "[mmjp_train] CRF supervised: cannot open %s\n", path);
    return 0;
  }
  file_iter_t it;
  memset(&it, 0, sizeof(it));
  it.f = f;
  it.max_line_bytes = max_line_bytes;
  it.max_sentence_cp = max_sentence_cp;
  it.skip_long_cp = 1;
  it.keep_chars = NULL;
  it.fallback_cp = (uint32_t)'?';

  while (1) {
    int rc = file_iter_readline(&it);
    if (rc == 0) break;
    if (rc < 0) {
      fprintf(stderr, "[mmjp_train] CRF supervised: readline failed rc=%d\n", rc);
      crf_dataset_free(out);
      fclose(f);
      free(it.buf);
      free(it.mapped);
      return 0;
    }
    if (it.len == 0) continue;
    uint8_t *cls = NULL;
    uint8_t *y = NULL;
    uint16_t n = 0;
    if (!crf_parse_segmented_line(it.buf, it.len, &cls, &y, &n, max_sentence_cp)) {
      continue; /* skip invalid/too long */
    }
    if (!crf_dataset_push(out, cls, y, n)) {
      free(cls);
      free(y);
      crf_dataset_free(out);
      fclose(f);
      free(it.buf);
      free(it.mapped);
      return 0;
    }
  }

  fclose(f);
  free(it.buf);
  free(it.mapped);
  return 1;
}

/*
 * 教師なしCRFデータ生成（LM-only Viterbi で擬似ラベルを作成）
 */
static int crf_dataset_from_lm_viterbi(const char *corpus_path,
                                       size_t max_line_bytes,
                                       size_t max_sentence_cp,
                                       const unilm_model_t *um,
                                       unilm_workspace_t *wk,
                                       int max_piece_len_cp,
                                       size_t limit_sentences,
                                       crf_dataset_t *out) {
  if (!corpus_path || !um || !wk || !out) return 0;
  memset(out, 0, sizeof(*out));

  FILE *f = fopen(corpus_path, "rb");
  if (!f) {
    fprintf(stderr, "[crf_unsup] cannot open %s\n", corpus_path);
    return 0;
  }

  file_iter_t it;
  memset(&it, 0, sizeof(it));
  it.f = f;
  it.max_line_bytes = max_line_bytes;
  it.max_sentence_cp = max_sentence_cp;
  it.skip_long_cp = 1;
  it.keep_chars = NULL;
  it.fallback_cp = (uint32_t)'?';

  /* Viterbi output buffer */
  size_t ids_cap = max_sentence_cp;
  uint32_t *ids = (uint32_t *)malloc(ids_cap * sizeof(uint32_t));
  if (!ids) {
    fclose(f);
    return 0;
  }

  size_t n_sent = 0;
  size_t n_read = 0;
  size_t n_viterbi_ok = 0;
  size_t n_viterbi_err = 0;
  while (n_sent < limit_sentences) {
    int r = file_iter_readline(&it);
    if (r == 0) break;
    if (r < 0) {
      fprintf(stderr, "[crf_unsup] readline failed rc=%d\n", r);
      break;
    }
    if (it.len == 0) continue;
    n_read++;

    const uint8_t *s = it.buf;
    size_t sl = it.len;

    /* Count codepoints first */
    size_t n_cp = 0;
    for (size_t pos = 0; pos < sl; ) {
      uint32_t cp = 0;
      size_t adv = 0;
      if (!utf8_decode1(s, sl, pos, &cp, &adv)) {
        pos++;
        continue;
      }
      n_cp++;
      pos += adv;
    }
    if (n_cp == 0 || n_cp > max_sentence_cp) continue;

    /* Run LM-only Viterbi */
    size_t out_n = 0;
    int rc = unilm_viterbi_tokenize(um, s, sl, max_piece_len_cp, wk, ids, ids_cap, &out_n);
    int use_char_fallback = (rc != UNILM_OK || out_n == 0);
    if (use_char_fallback) {
      n_viterbi_err++;
    } else {
      n_viterbi_ok++;
    }

    /* Allocate cls and y */
    uint8_t *cls = (uint8_t *)malloc(n_cp);
    uint8_t *y = (uint8_t *)malloc(n_cp);
    if (!cls || !y) {
      free(cls);
      free(y);
      continue;
    }

    if (use_char_fallback) {
      /* Fallback: character-level tokenization (all chars are boundaries) */
      memset(y, 1, n_cp);  /* all boundaries */
      size_t byte_pos = 0;
      size_t cp_idx = 0;
      while (byte_pos < sl && cp_idx < n_cp) {
        uint32_t cp = 0;
        size_t adv = 0;
        if (!utf8_decode1(s, sl, byte_pos, &cp, &adv)) {
          byte_pos++;
          continue;
        }
        cls[cp_idx] = npycrf_char_class_cp(NULL, cp);
        cp_idx++;
        byte_pos += adv;
      }
    } else {
      /* Build boundary array from Viterbi output */
      memset(y, 0, n_cp);
      size_t cp_idx = 0;
      size_t piece_idx = 0;
      size_t byte_pos = 0;

      while (byte_pos < sl && cp_idx < n_cp && piece_idx < out_n) {
        /* Get current piece length */
        size_t piece_bytes = 0;
        const uint8_t *piece_str = unilm_model_piece_bytes(um, ids[piece_idx], &piece_bytes);
        (void)piece_str;

        /* Mark boundary at start of this piece */
        y[cp_idx] = 1;

        /* Consume codepoints for this piece */
        size_t consumed_bytes = 0;
        while (consumed_bytes < piece_bytes && byte_pos < sl && cp_idx < n_cp) {
          uint32_t cp = 0;
          size_t adv = 0;
          if (!utf8_decode1(s, sl, byte_pos, &cp, &adv)) {
            byte_pos++;
            continue;
          }
          cls[cp_idx] = npycrf_char_class_cp(NULL, cp);
          cp_idx++;
          byte_pos += adv;
          consumed_bytes += adv;
        }
        piece_idx++;
      }

      /* Fill remaining cls if any */
      while (byte_pos < sl && cp_idx < n_cp) {
        uint32_t cp = 0;
        size_t adv = 0;
        if (!utf8_decode1(s, sl, byte_pos, &cp, &adv)) {
          byte_pos++;
          continue;
        }
        cls[cp_idx] = npycrf_char_class_cp(NULL, cp);
        y[cp_idx] = 1;  /* mark as boundary */
        cp_idx++;
        byte_pos += adv;
      }
    }

    /* Enforce y[0] = 1 */
    if (n_cp > 0) y[0] = 1;

    if (!crf_dataset_push(out, cls, y, (uint16_t)n_cp)) {
      free(cls);
      free(y);
      continue;
    }
    n_sent++;
  }

  free(ids);
  fclose(f);
  free(it.buf);
  free(it.mapped);

  fprintf(stderr, "[crf_unsup] read=%zu viterbi_ok=%zu viterbi_err=%zu pushed=%zu\n",
          n_read, n_viterbi_ok, n_viterbi_err, out->n);
  return out->n > 0 ? 1 : 0;
}

static double logsumexp2(double a, double b) {
  if (isinf(a) && a < 0) return b;
  if (isinf(b) && b < 0) return a;
  double m = (a > b) ? a : b;
  return m + log(exp(a - m) + exp(b - m));
}

static double crf_emit_score_one(const crf_table_t *tbl, const double *feat_w,
                                 uint8_t label, uint8_t prev_c, uint8_t cur_c, uint8_t next_c) {
  double s = 0.0;
  /* template 0: cur */
  {
    uint32_t key = npycrf_feat_key(0, label, cur_c, 0);
    int idx = crf_table_find_idx(tbl, key);
    if (idx >= 0) s += feat_w[(size_t)idx];
  }
  /* template 1: prev */
  {
    uint32_t key = npycrf_feat_key(1, label, prev_c, 0);
    int idx = crf_table_find_idx(tbl, key);
    if (idx >= 0) s += feat_w[(size_t)idx];
  }
  /* template 2: next */
  {
    uint32_t key = npycrf_feat_key(2, label, next_c, 0);
    int idx = crf_table_find_idx(tbl, key);
    if (idx >= 0) s += feat_w[(size_t)idx];
  }
  /* template 3: prev-cur */
  {
    uint32_t key = npycrf_feat_key(3, label, prev_c, cur_c);
    int idx = crf_table_find_idx(tbl, key);
    if (idx >= 0) s += feat_w[(size_t)idx];
  }
  /* template 4: cur-next */
  {
    uint32_t key = npycrf_feat_key(4, label, cur_c, next_c);
    int idx = crf_table_find_idx(tbl, key);
    if (idx >= 0) s += feat_w[(size_t)idx];
  }
  return s;
}

static void crf_add_feat_grad(const crf_table_t *tbl, double *grad_feat,
                              double coeff,
                              uint8_t label, uint8_t prev_c, uint8_t cur_c, uint8_t next_c) {
  /* mirror crf_emit_score_one */
  uint32_t key;
  int idx;
  key = npycrf_feat_key(0, label, cur_c, 0);
  idx = crf_table_find_idx(tbl, key);
  if (idx >= 0) grad_feat[(size_t)idx] += coeff;

  key = npycrf_feat_key(1, label, prev_c, 0);
  idx = crf_table_find_idx(tbl, key);
  if (idx >= 0) grad_feat[(size_t)idx] += coeff;

  key = npycrf_feat_key(2, label, next_c, 0);
  idx = crf_table_find_idx(tbl, key);
  if (idx >= 0) grad_feat[(size_t)idx] += coeff;

  key = npycrf_feat_key(3, label, prev_c, cur_c);
  idx = crf_table_find_idx(tbl, key);
  if (idx >= 0) grad_feat[(size_t)idx] += coeff;

  key = npycrf_feat_key(4, label, cur_c, next_c);
  idx = crf_table_find_idx(tbl, key);
  if (idx >= 0) grad_feat[(size_t)idx] += coeff;
}

/* 2-label linear-chain CRF training for boundary labels.
 *
 * y[i] = 1 (start) or 0 (internal).
 * Constraints:
 *   - y[0] = 1
 *   - implicit EOS label is 1 (adds a final transition y[n-1] -> 1)
 */
static int crf_train_supervised(const crf_dataset_t *ds,
                                const crf_table_t *tbl,
                                double *feat_w,
                                double *trans00, double *trans01, double *trans10, double *trans11,
                                int epochs,
                                double lr,
                                double l2) {
  if (!ds || ds->n == 0 || !tbl || !feat_w || !trans00 || !trans01 || !trans10 || !trans11) return 0;
  if (epochs <= 0) epochs = 1;
  if (lr <= 0) lr = 0.05;
  if (l2 < 0) l2 = 0.0;

  const size_t nfeat = (size_t)tbl->n;
  double *grad_feat = (double *)calloc(nfeat, sizeof(double));
  if (!grad_feat) return 0;

  for (int ep = 0; ep < epochs; ep++) {
    memset(grad_feat, 0, nfeat * sizeof(double));
    double g_t00 = 0.0, g_t01 = 0.0, g_t10 = 0.0, g_t11 = 0.0;
    double total_ll = 0.0;

    for (size_t si = 0; si < ds->n; si++) {
      const crf_sent_t *s = &ds->s[si];
      uint16_t n = s->n;
      if (n == 0) continue;

      /* emission scores */
      double *e0 = (double *)malloc((size_t)n * sizeof(double));
      double *e1 = (double *)malloc((size_t)n * sizeof(double));
      if (!e0 || !e1) {
        free(e0);
        free(e1);
        free(grad_feat);
        return 0;
      }
      for (uint16_t i = 0; i < n; i++) {
        uint8_t prev_c = (i == 0) ? MMJP_CC_BOS : s->cls[i - 1];
        uint8_t cur_c = s->cls[i];
        uint8_t next_c = (i + 1 == n) ? MMJP_CC_EOS : s->cls[i + 1];
        e0[i] = crf_emit_score_one(tbl, feat_w, 0, prev_c, cur_c, next_c);
        e1[i] = crf_emit_score_one(tbl, feat_w, 1, prev_c, cur_c, next_c);
      }

      /* forward (log-space) */
      double *a0 = (double *)malloc((size_t)n * sizeof(double));
      double *a1 = (double *)malloc((size_t)n * sizeof(double));
      double *b0 = (double *)malloc((size_t)n * sizeof(double));
      double *b1 = (double *)malloc((size_t)n * sizeof(double));
      if (!a0 || !a1 || !b0 || !b1) {
        free(e0);
        free(e1);
        free(a0);
        free(a1);
        free(b0);
        free(b1);
        free(grad_feat);
        return 0;
      }

      a0[0] = -INFINITY;
      a1[0] = e1[0]; /* y0 fixed to 1; bos_to1 is constant -> omitted */
      for (uint16_t i = 1; i < n; i++) {
        /* NOTE: MMJP の CRF 遷移は (label: word-start=1, inside=0)
         *   0->0: trans00
         *   1->0: trans01
         *   0->1: trans10
         *   1->1: trans11
         */
        a0[i] = e0[i] + logsumexp2(a0[i - 1] + (*trans00), a1[i - 1] + (*trans01));
        a1[i] = e1[i] + logsumexp2(a0[i - 1] + (*trans10), a1[i - 1] + (*trans11));
      }
      double logZ = logsumexp2(a0[n - 1] + (*trans10), a1[n - 1] + (*trans11)); /* EOS label fixed to 1 */

      /* backward */
      b0[n - 1] = (*trans10);
      b1[n - 1] = (*trans11);
      for (int i = (int)n - 2; i >= 0; i--) {
        b0[i] = logsumexp2((*trans00) + e0[i + 1] + b0[i + 1], (*trans10) + e1[i + 1] + b1[i + 1]);
        b1[i] = logsumexp2((*trans01) + e0[i + 1] + b0[i + 1], (*trans11) + e1[i + 1] + b1[i + 1]);
      }

      /* empirical score */
      double st = e1[0];
      for (uint16_t i = 1; i < n; i++) {
        uint8_t yp = s->y[i - 1];
        uint8_t yc = s->y[i];
        if (yp == 0 && yc == 0) st += (*trans00);
        else if (yp == 0 && yc == 1) st += (*trans10);
        else if (yp == 1 && yc == 0) st += (*trans01);
        else st += (*trans11);
        st += (yc ? e1[i] : e0[i]);
      }
      /* final transition to EOS=1 */
      if (s->y[n - 1] == 0) st += (*trans10);
      else st += (*trans11);

      total_ll += (st - logZ);

      /* gradients: transitions */
      /* expected transition counts */
      double exp_t00 = 0.0, exp_t01 = 0.0, exp_t10 = 0.0, exp_t11 = 0.0;
      for (uint16_t i = 1; i < n; i++) {
        /* pair marginals */
        double p00 = exp(a0[i - 1] + (*trans00) + e0[i] + b0[i] - logZ);
        double p01 = exp(a0[i - 1] + (*trans10) + e1[i] + b1[i] - logZ);
        double p10 = exp(a1[i - 1] + (*trans01) + e0[i] + b0[i] - logZ);
        double p11 = exp(a1[i - 1] + (*trans11) + e1[i] + b1[i] - logZ);
        exp_t00 += p00;
        exp_t10 += p01; /* 0->1 */
        exp_t01 += p10; /* 1->0 */
        exp_t11 += p11;
      }
      /* final transition to EOS=1 */
      exp_t10 += exp(a0[n - 1] + (*trans10) - logZ);
      exp_t11 += exp(a1[n - 1] + (*trans11) - logZ);

      /* empirical transition counts */
      double emp_t00 = 0.0, emp_t01 = 0.0, emp_t10 = 0.0, emp_t11 = 0.0;
      for (uint16_t i = 1; i < n; i++) {
        uint8_t yp = s->y[i - 1];
        uint8_t yc = s->y[i];
        if (yp == 0 && yc == 0) emp_t00 += 1.0;
        else if (yp == 0 && yc == 1) emp_t10 += 1.0;
        else if (yp == 1 && yc == 0) emp_t01 += 1.0;
        else emp_t11 += 1.0;
      }
      if (s->y[n - 1] == 0) emp_t10 += 1.0;
      else emp_t11 += 1.0;

      g_t00 += (emp_t00 - exp_t00);
      g_t01 += (emp_t01 - exp_t01);
      g_t10 += (emp_t10 - exp_t10);
      g_t11 += (emp_t11 - exp_t11);

      /* gradients: features */
      for (uint16_t i = 0; i < n; i++) {
        uint8_t prev_c = (i == 0) ? MMJP_CC_BOS : s->cls[i - 1];
        uint8_t cur_c = s->cls[i];
        uint8_t next_c = (i + 1 == n) ? MMJP_CC_EOS : s->cls[i + 1];
        /* empirical */
        crf_add_feat_grad(tbl, grad_feat, 1.0, s->y[i], prev_c, cur_c, next_c);
        /* expected */
        double p0 = exp(a0[i] + b0[i] - logZ);
        double p1 = exp(a1[i] + b1[i] - logZ);
        crf_add_feat_grad(tbl, grad_feat, -p0, 0, prev_c, cur_c, next_c);
        crf_add_feat_grad(tbl, grad_feat, -p1, 1, prev_c, cur_c, next_c);
      }

      free(e0);
      free(e1);
      free(a0);
      free(a1);
      free(b0);
      free(b1);
    }

    /* L2 */
    if (l2 > 0) {
      g_t00 -= l2 * (*trans00);
      g_t01 -= l2 * (*trans01);
      g_t10 -= l2 * (*trans10);
      g_t11 -= l2 * (*trans11);
      for (size_t i = 0; i < nfeat; i++) grad_feat[i] -= l2 * feat_w[i];
    }

    /* update (average by total positions for stability) */
    double scale = (ds->total_pos > 0) ? (1.0 / (double)ds->total_pos) : 1.0;
    double step = lr * scale;
    *trans00 += step * g_t00;
    *trans01 += step * g_t01;
    *trans10 += step * g_t10;
    *trans11 += step * g_t11;
    for (size_t i = 0; i < nfeat; i++) feat_w[i] += step * grad_feat[i];

    printf("[mmjp_train] CRF supervised ep=%d/%d ll=%.3f (trans00=%.3f trans01=%.3f trans10=%.3f trans11=%.3f)\n",
           ep + 1, epochs, total_ll, *trans00, *trans01, *trans10, *trans11);
  }

  free(grad_feat);
  return 1;
}

/* =====================
 * CRF supervised training (L-BFGS)
 * ===================== */

/* L-BFGS implementation (no external deps). Suitable for small supervised datasets. */

static double vec_dot(const double *a, const double *b, size_t n) {
  double s = 0.0;
  for (size_t i = 0; i < n; i++) s += a[i] * b[i];
  return s;
}

static double vec_norm2(const double *a, size_t n) {
  return sqrt(vec_dot(a, a, n));
}

static void vec_copy(double *dst, const double *src, size_t n) {
  memcpy(dst, src, n * sizeof(double));
}

static void vec_axpy(double *y, double a, const double *x, size_t n) {
  for (size_t i = 0; i < n; i++) y[i] += a * x[i];
}

static void vec_scale(double *x, double a, size_t n) {
  for (size_t i = 0; i < n; i++) x[i] *= a;
}

typedef struct {
  const crf_dataset_t *ds;
  const crf_table_t *tbl;
  size_t nfeat;
  double l2;

  /* reusable workspace (max sentence length) */
  uint16_t max_n;
  double *e0, *e1, *a0, *a1, *b0, *b1;

  /* last evaluation (unscaled) */
  double last_ll;
  double last_pen;
} crf_eval_ctx_t;

static double crf_eval_obj_grad_min(const double *x, double *g, void *vctx) {
  crf_eval_ctx_t *ctx = (crf_eval_ctx_t *)vctx;
  const crf_dataset_t *ds = ctx->ds;
  const crf_table_t *tbl = ctx->tbl;
  const size_t nfeat = ctx->nfeat;

  const double *feat_w = x;
  double trans00 = x[nfeat + 0];
  double trans01 = x[nfeat + 1];
  double trans10 = x[nfeat + 2];
  double trans11 = x[nfeat + 3];

  /* g will temporarily hold gradient of maximization objective:
   *   J = ll - (l2/2)||w||^2
   * then converted to minimization grad of f = -J / total_pos.
   */
  memset(g, 0, (nfeat + 4u) * sizeof(double));

  double total_ll = 0.0;

  for (size_t si = 0; si < ds->n; si++) {
    const crf_sent_t *s = &ds->s[si];
    uint16_t n = s->n;
    if (n == 0) continue;
    if (n > ctx->max_n) n = ctx->max_n;

    /* emission scores */
    for (uint16_t i = 0; i < n; i++) {
      uint8_t prev_c = (i == 0) ? MMJP_CC_BOS : s->cls[i - 1];
      uint8_t cur_c = s->cls[i];
      uint8_t next_c = (i + 1 == n) ? MMJP_CC_EOS : s->cls[i + 1];
      ctx->e0[i] = crf_emit_score_one(tbl, feat_w, 0, prev_c, cur_c, next_c);
      ctx->e1[i] = crf_emit_score_one(tbl, feat_w, 1, prev_c, cur_c, next_c);
    }

    /* forward (log-space) */
    ctx->a0[0] = -INFINITY;
    ctx->a1[0] = ctx->e1[0]; /* y0 fixed to 1; bos_to1 is constant -> omitted */
    for (uint16_t i = 1; i < n; i++) {
      ctx->a0[i] = ctx->e0[i] + logsumexp2(ctx->a0[i - 1] + trans00, ctx->a1[i - 1] + trans01);
      ctx->a1[i] = ctx->e1[i] + logsumexp2(ctx->a0[i - 1] + trans10, ctx->a1[i - 1] + trans11);
    }
    double logZ = logsumexp2(ctx->a0[n - 1] + trans10, ctx->a1[n - 1] + trans11); /* EOS label fixed to 1 */

    /* backward */
    ctx->b0[n - 1] = trans10;
    ctx->b1[n - 1] = trans11;
    for (int i = (int)n - 2; i >= 0; i--) {
      ctx->b0[i] = logsumexp2(trans00 + ctx->e0[i + 1] + ctx->b0[i + 1],
                              trans10 + ctx->e1[i + 1] + ctx->b1[i + 1]);
      ctx->b1[i] = logsumexp2(trans01 + ctx->e0[i + 1] + ctx->b0[i + 1],
                              trans11 + ctx->e1[i + 1] + ctx->b1[i + 1]);
    }

    /* empirical score */
    double st = ctx->e1[0];
    for (uint16_t i = 1; i < n; i++) {
      uint8_t yp = s->y[i - 1];
      uint8_t yc = s->y[i];
      if (yp == 0 && yc == 0) st += trans00;
      else if (yp == 0 && yc == 1) st += trans10;
      else if (yp == 1 && yc == 0) st += trans01;
      else st += trans11;
      st += (yc ? ctx->e1[i] : ctx->e0[i]);
    }
    /* final transition to EOS=1 */
    if (s->y[n - 1] == 0) st += trans10;
    else st += trans11;

    total_ll += (st - logZ);

    /* expected transition counts */
    double exp_t00 = 0.0, exp_t01 = 0.0, exp_t10 = 0.0, exp_t11 = 0.0;
    for (uint16_t i = 1; i < n; i++) {
      double p00 = exp(ctx->a0[i - 1] + trans00 + ctx->e0[i] + ctx->b0[i] - logZ);
      double p01 = exp(ctx->a0[i - 1] + trans10 + ctx->e1[i] + ctx->b1[i] - logZ);
      double p10 = exp(ctx->a1[i - 1] + trans01 + ctx->e0[i] + ctx->b0[i] - logZ);
      double p11 = exp(ctx->a1[i - 1] + trans11 + ctx->e1[i] + ctx->b1[i] - logZ);
      exp_t00 += p00;
      exp_t10 += p01; /* 0->1 */
      exp_t01 += p10; /* 1->0 */
      exp_t11 += p11;
    }
    /* final transition to EOS=1 */
    exp_t10 += exp(ctx->a0[n - 1] + trans10 - logZ);
    exp_t11 += exp(ctx->a1[n - 1] + trans11 - logZ);

    /* empirical transition counts */
    double emp_t00 = 0.0, emp_t01 = 0.0, emp_t10 = 0.0, emp_t11 = 0.0;
    for (uint16_t i = 1; i < n; i++) {
      uint8_t yp = s->y[i - 1];
      uint8_t yc = s->y[i];
      if (yp == 0 && yc == 0) emp_t00 += 1.0;
      else if (yp == 0 && yc == 1) emp_t10 += 1.0;
      else if (yp == 1 && yc == 0) emp_t01 += 1.0;
      else emp_t11 += 1.0;
    }
    if (s->y[n - 1] == 0) emp_t10 += 1.0;
    else emp_t11 += 1.0;

    g[nfeat + 0] += (emp_t00 - exp_t00);
    g[nfeat + 1] += (emp_t01 - exp_t01);
    g[nfeat + 2] += (emp_t10 - exp_t10);
    g[nfeat + 3] += (emp_t11 - exp_t11);

    /* gradients: features */
    for (uint16_t i = 0; i < n; i++) {
      uint8_t prev_c = (i == 0) ? MMJP_CC_BOS : s->cls[i - 1];
      uint8_t cur_c = s->cls[i];
      uint8_t next_c = (i + 1 == n) ? MMJP_CC_EOS : s->cls[i + 1];

      /* empirical */
      crf_add_feat_grad(tbl, g, 1.0, s->y[i], prev_c, cur_c, next_c);

      /* expected */
      double p0 = exp(ctx->a0[i] + ctx->b0[i] - logZ);
      double p1 = exp(ctx->a1[i] + ctx->b1[i] - logZ);
      crf_add_feat_grad(tbl, g, -p0, 0, prev_c, cur_c, next_c);
      crf_add_feat_grad(tbl, g, -p1, 1, prev_c, cur_c, next_c);
    }
  }

  /* L2 (on maximization objective J) */
  double w2 = 0.0;
  if (ctx->l2 > 0) {
    for (size_t i = 0; i < nfeat; i++) w2 += feat_w[i] * feat_w[i];
    w2 += trans00 * trans00 + trans01 * trans01 + trans10 * trans10 + trans11 * trans11;

    g[nfeat + 0] -= ctx->l2 * trans00;
    g[nfeat + 1] -= ctx->l2 * trans01;
    g[nfeat + 2] -= ctx->l2 * trans10;
    g[nfeat + 3] -= ctx->l2 * trans11;
    for (size_t i = 0; i < nfeat; i++) g[i] -= ctx->l2 * feat_w[i];
  } else {
    for (size_t i = 0; i < nfeat; i++) w2 += feat_w[i] * feat_w[i];
    w2 += trans00 * trans00 + trans01 * trans01 + trans10 * trans10 + trans11 * trans11;
  }

  double pen = 0.5 * ctx->l2 * w2;

  ctx->last_ll = total_ll;
  ctx->last_pen = pen;

  double scale = (ds->total_pos > 0) ? (1.0 / (double)ds->total_pos) : 1.0;
  double J = total_ll - pen;
  double f = -J * scale;

  /* Convert to minimization gradient: grad(f) = -grad(J) / total_pos */
  for (size_t i = 0; i < nfeat + 4u; i++) g[i] = -g[i] * scale;

  return f;
}

static int lbfgs_minimize(double *x, size_t n,
                          int max_iter,
                          int m_hist,
                          double tol,
                          int ls_max,
                          double (*eval)(const double *x, double *g, void *ctx),
                          void *ctx) {
  if (!x || n == 0 || !eval) return 0;
  if (max_iter <= 0) max_iter = 1;
  if (m_hist <= 0) m_hist = 8;
  if (m_hist > 32) m_hist = 32;
  if (tol <= 0) tol = 1e-5;
  if (ls_max <= 0) ls_max = 20;

  double *g = (double *)malloc(n * sizeof(double));
  double *g_new = (double *)malloc(n * sizeof(double));
  double *x_new = (double *)malloc(n * sizeof(double));
  double *d = (double *)malloc(n * sizeof(double));
  double *q = (double *)malloc(n * sizeof(double));
  double *r = (double *)malloc(n * sizeof(double));
  double *alpha = (double *)malloc((size_t)m_hist * sizeof(double));
  double *rho = (double *)malloc((size_t)m_hist * sizeof(double));
  double *s_hist = (double *)malloc((size_t)m_hist * n * sizeof(double));
  double *y_hist = (double *)malloc((size_t)m_hist * n * sizeof(double));
  if (!g || !g_new || !x_new || !d || !q || !r || !alpha || !rho || !s_hist || !y_hist) {
    free(g);
    free(g_new);
    free(x_new);
    free(d);
    free(q);
    free(r);
    free(alpha);
    free(rho);
    free(s_hist);
    free(y_hist);
    return 0;
  }

  int hist_count = 0;
  int hist_start = 0;

  double f = eval(x, g, ctx);

  for (int it = 0; it < max_iter; it++) {
    double gnorm = vec_norm2(g, n);
    if (gnorm < tol) {
      printf("[mmjp_train] CRF lbfgs converged it=%d grad_norm=%.6g\n", it, gnorm);
      break;
    }

    /* compute direction via two-loop recursion */
    vec_copy(q, g, n);
    for (int i = hist_count - 1; i >= 0; i--) {
      int idx = (hist_start + i) % m_hist;
      const double *s = s_hist + (size_t)idx * n;
      const double *y = y_hist + (size_t)idx * n;
      double a = rho[idx] * vec_dot(s, q, n);
      alpha[i] = a;
      /* q = q - a*y */
      for (size_t j = 0; j < n; j++) q[j] -= a * y[j];
    }

    double H0 = 1.0;
    if (hist_count > 0) {
      int idx_last = (hist_start + hist_count - 1) % m_hist;
      const double *s = s_hist + (size_t)idx_last * n;
      const double *y = y_hist + (size_t)idx_last * n;
      double sy = vec_dot(s, y, n);
      double yy = vec_dot(y, y, n);
      if (yy > 0) H0 = sy / yy;
    }

    vec_copy(r, q, n);
    vec_scale(r, H0, n);

    for (int i = 0; i < hist_count; i++) {
      int idx = (hist_start + i) % m_hist;
      const double *s = s_hist + (size_t)idx * n;
      const double *y = y_hist + (size_t)idx * n;
      double b = rho[idx] * vec_dot(y, r, n);
      double a = alpha[i];
      /* r = r + s*(a-b) */
      vec_axpy(r, (a - b), s, n);
    }

    /* d = -r */
    vec_copy(d, r, n);
    vec_scale(d, -1.0, n);

    double gtd = vec_dot(g, d, n);
    if (!(gtd < 0.0)) {
      /* not a descent direction -> reset */
      vec_copy(d, g, n);
      vec_scale(d, -1.0, n);
      gtd = vec_dot(g, d, n);
      hist_count = 0;
      hist_start = 0;
    }

    /* backtracking line search (Armijo) */
    double t = 1.0;
    const double c1 = 1e-4;
    int accepted = 0;
    double f_new = f;

    for (int ls = 0; ls < ls_max; ls++) {
      vec_copy(x_new, x, n);
      vec_axpy(x_new, t, d, n);

      f_new = eval(x_new, g_new, ctx);
      if (f_new <= f + c1 * t * gtd) {
        accepted = 1;
        break;
      }
      t *= 0.5;
      if (t < 1e-20) break;
    }

    if (!accepted) {
      printf("[mmjp_train] CRF lbfgs line-search failed (it=%d). stop.\n", it);
      break;
    }

    /* s = x_new - x, y = g_new - g */
    int store_idx;
    if (hist_count < m_hist) {
      store_idx = (hist_start + hist_count) % m_hist;
      hist_count++;
    } else {
      store_idx = hist_start;
      hist_start = (hist_start + 1) % m_hist;
    }
    double *s = s_hist + (size_t)store_idx * n;
    double *y = y_hist + (size_t)store_idx * n;

    for (size_t j = 0; j < n; j++) {
      s[j] = x_new[j] - x[j];
      y[j] = g_new[j] - g[j];
    }
    double ys = vec_dot(y, s, n);
    if (ys > 1e-12) {
      rho[store_idx] = 1.0 / ys;
    } else {
      /* skip update if numerically bad */
      hist_count = 0;
      hist_start = 0;
    }

    vec_copy(x, x_new, n);
    vec_copy(g, g_new, n);
    f = f_new;

    printf("[mmjp_train] CRF lbfgs it=%d/%d f=%.6f grad_norm=%.6g step=%.3g\n",
           it + 1, max_iter, f, vec_norm2(g, n), t);
  }

  free(g);
  free(g_new);
  free(x_new);
  free(d);
  free(q);
  free(r);
  free(alpha);
  free(rho);
  free(s_hist);
  free(y_hist);
  return 1;
}

static int crf_train_supervised_lbfgs(const crf_dataset_t *ds,
                                      const crf_table_t *tbl,
                                      double *feat_w,
                                      double *trans00, double *trans01, double *trans10, double *trans11,
                                      int max_iter,
                                      double l2,
                                      int m_hist,
                                      double tol) {
  if (!ds || ds->n == 0 || !tbl || !feat_w || !trans00 || !trans01 || !trans10 || !trans11) return 0;

  size_t nfeat = (size_t)tbl->n;
  size_t dim = nfeat + 4u;

  /* determine max sentence length for reusable workspace */
  uint16_t max_n = 0;
  for (size_t i = 0; i < ds->n; i++) {
    if (ds->s[i].n > max_n) max_n = ds->s[i].n;
  }
  if (max_n == 0) return 0;

  crf_eval_ctx_t ctx;
  memset(&ctx, 0, sizeof(ctx));
  ctx.ds = ds;
  ctx.tbl = tbl;
  ctx.nfeat = nfeat;
  ctx.l2 = l2;
  ctx.max_n = max_n;

  ctx.e0 = (double *)malloc((size_t)max_n * sizeof(double));
  ctx.e1 = (double *)malloc((size_t)max_n * sizeof(double));
  ctx.a0 = (double *)malloc((size_t)max_n * sizeof(double));
  ctx.a1 = (double *)malloc((size_t)max_n * sizeof(double));
  ctx.b0 = (double *)malloc((size_t)max_n * sizeof(double));
  ctx.b1 = (double *)malloc((size_t)max_n * sizeof(double));

  if (!ctx.e0 || !ctx.e1 || !ctx.a0 || !ctx.a1 || !ctx.b0 || !ctx.b1) {
    free(ctx.e0); free(ctx.e1); free(ctx.a0); free(ctx.a1); free(ctx.b0); free(ctx.b1);
    return 0;
  }

  double *x = (double *)malloc(dim * sizeof(double));
  if (!x) {
    free(ctx.e0); free(ctx.e1); free(ctx.a0); free(ctx.a1); free(ctx.b0); free(ctx.b1);
    return 0;
  }
  for (size_t i = 0; i < nfeat; i++) x[i] = feat_w[i];
  x[nfeat + 0] = *trans00;
  x[nfeat + 1] = *trans01;
  x[nfeat + 2] = *trans10;
  x[nfeat + 3] = *trans11;

  printf("[mmjp_train] CRF supervised (lbfgs): iter=%d m=%d tol=%.2g l2=%.2g\n",
         max_iter, m_hist, tol, l2);

  int ok = lbfgs_minimize(x, dim, max_iter, m_hist, tol, 20, crf_eval_obj_grad_min, &ctx);

  for (size_t i = 0; i < nfeat; i++) feat_w[i] = x[i];
  *trans00 = x[nfeat + 0];
  *trans01 = x[nfeat + 1];
  *trans10 = x[nfeat + 2];
  *trans11 = x[nfeat + 3];

  printf("[mmjp_train] CRF supervised (lbfgs) done: trans00=%.4f trans01=%.4f trans10=%.4f trans11=%.4f\n",
         *trans00, *trans01, *trans10, *trans11);

  free(x);
  free(ctx.e0); free(ctx.e1); free(ctx.a0); free(ctx.a1); free(ctx.b0); free(ctx.b1);
  return ok;
}

/* =====================
 * Candidate extraction via suffix array
 * ===================== */
static int mmjp_bytes_contains(const char *s, size_t len, const uint8_t *pat, size_t pat_len) {
  if (!s || len == 0 || !pat || pat_len == 0) return 0;
  if (pat_len > len) return 0;
  for (size_t i = 0; i + pat_len <= len; i++) {
    if (memcmp(s + i, pat, pat_len) == 0) return 1;
  }
  return 0;
}



static int collect_top_ngrams(const uint8_t *text, size_t text_len,
                             int max_piece_len_cp,
                             size_t cand_total,
                             uint32_t min_count,
                             const uint8_t *fb, size_t fb_len,
                             cand_t **out_cands,
                             size_t *out_n) {
  if (!text || text_len == 0 || !out_cands || !out_n) return 0;

  unsigned build_flags = SA_BUILD_SKIP_ASCII_SPACE | SA_BUILD_SKIP_ASCII_PUNCT;
  size_t starts = sa_utf8_count_starts(text, text_len, build_flags);
  if (starts == 0) {
    /* テキストが ASCII 句読点（例: fallback='?'）に偏っている場合の救済 */
    build_flags = SA_BUILD_SKIP_ASCII_SPACE;
    starts = sa_utf8_count_starts(text, text_len, build_flags);
  }
  if (starts == 0) {
    build_flags = SA_BUILD_DEFAULT;
    starts = sa_utf8_count_starts(text, text_len, build_flags);
  }
  if (starts == 0) {
    fprintf(stderr, "[mmjp_train] suffix-array: no valid starts (text too small or mostly skipped chars)\n");
    return 0;
  }

  const size_t sa_bytes = starts * sizeof(sa_idx_t);
  fprintf(stderr, "[mmjp_train] suffix-array: starts=%zu (%.1f MB), flags=0x%X\n",
          starts, (double)sa_bytes / (1024.0 * 1024.0), (unsigned)build_flags);

  sa_idx_t *sa = (sa_idx_t *)malloc(sa_bytes);
  if (!sa) {
    fprintf(stderr, "[mmjp_train] suffix-array: oom for sa (%zu bytes)\n", sa_bytes);
    return 0;
  }
  size_t built = sa_utf8_build(sa, starts, text, text_len, build_flags);
  if (built == 0) {
    fprintf(stderr, "[mmjp_train] suffix-array: build failed. Hint: compile suffix_array with -DSA_SORT_DYNAMIC_STACK=1 or increase SA_SORT_STACK_MAX, or reduce --sample_bytes.\n");
    free(sa);
    return 0;
  }
  starts = built;

  int n_min = 2;
  int n_max = (max_piece_len_cp > 1) ? max_piece_len_cp : 2;
  int n_len = (n_max - n_min + 1);
  size_t per_len = (cand_total > 0 && n_len > 0) ? (cand_total / (size_t)n_len) : 0;
  if (per_len < 512) per_len = 512;

  cand_t *all = NULL;
  size_t all_n = 0, all_cap = 0;

  char last[128];
  char cur[128];

  for (int ncp = n_min; ncp <= n_max; ncp++) {
    cand_heap_t heap;
    if (!heap_init(&heap, per_len)) {
      free(sa);
      return 0;
    }
    last[0] = '\0';
    uint32_t run = 0;

    for (size_t i = 0; i < starts; i++) {
      size_t start_pos = sa[i];
      size_t w = sa_utf8_copy_prefix_n(text, text_len, start_pos, (size_t)ncp, cur, sizeof(cur), SA_BUILD_DEFAULT);
      if (w == 0) continue;
      if (!is_good_piece_bytes(cur, w)) continue;
      if (fb && fb_len > 0 && mmjp_bytes_contains(cur, w, fb, fb_len)) continue;
      if (utf8_count_cp((const uint8_t *)cur, w) < (size_t)ncp) continue;

      if (run == 0) {
        memcpy(last, cur, w + 1);
        run = 1;
        continue;
      }
      if (strcmp(cur, last) == 0) {
        run++;
      } else {
        if (run >= min_count) {
          size_t blen = strlen(last);
          (void)heap_push_topk(&heap, run, last, (uint16_t)blen, (uint16_t)ncp);
        }
        /* restart */
        memcpy(last, cur, w + 1);
        run = 1;
      }
    }
    if (run >= min_count && last[0] != '\0') {
      size_t blen = strlen(last);
      (void)heap_push_topk(&heap, run, last, (uint16_t)blen, (uint16_t)ncp);
    }

    /* move heap entries to all */
    if (heap.n > 0) {
      if (all_n + heap.n > all_cap) {
        size_t nc = all_cap ? all_cap * 2 : 4096;
        while (nc < all_n + heap.n) nc *= 2;
        cand_t *na = (cand_t *)realloc(all, nc * sizeof(cand_t));
        if (!na) {
          heap_free(&heap);
          free(sa);
          return 0;
        }
        all = na;
        all_cap = nc;
      }
      for (size_t i = 0; i < heap.n; i++) {
        all[all_n++] = heap.a[i];
        /* ownership transferred */
        heap.a[i].s = NULL;
      }
    }
    heap_free(&heap);
  }

  free(sa);

  /* sort all and keep top cand_total */
  qsort(all, all_n, sizeof(cand_t), cmp_cand_desc);
  if (cand_total > 0 && all_n > cand_total) {
    for (size_t i = cand_total; i < all_n; i++) cand_free(&all[i]);
    all_n = cand_total;
  }

  *out_cands = all;
  *out_n = all_n;
  return 1;
}

/* =====================
 * Export vocabulary selection
 * ===================== */

typedef struct {
  uint32_t id;
  double p;
} idscore_t;

typedef struct {
  uint32_t cp;
  uint32_t cnt;
} cpair_t;

static int cmp_idscore_desc(const void *a, const void *b) {
  const idscore_t *x = (const idscore_t *)a;
  const idscore_t *y = (const idscore_t *)b;
  if (x->p < y->p) return 1;
  if (x->p > y->p) return -1;
  return (int)x->id - (int)y->id;
}

static int cmp_cpair_desc(const void *a, const void *b) {
  const cpair_t *x = (const cpair_t *)a;
  const cpair_t *y = (const cpair_t *)b;
  if (x->cnt < y->cnt) return 1;
  if (x->cnt > y->cnt) return -1;
  return (int)x->cp - (int)y->cp;
}

/* =====================
 * cc_ranges parser
 * ===================== */

/* Compare for sorting by lo (ascending) */
static int cmp_cc_range_lo(const void *a, const void *b) {
  const npycrf_cc_range_t *x = (const npycrf_cc_range_t *)a;
  const npycrf_cc_range_t *y = (const npycrf_cc_range_t *)b;
  if (x->lo < y->lo) return -1;
  if (x->lo > y->lo) return 1;
  return 0;
}

/*
 * Parse a cc_ranges file.
 * Format per line: start end class_id
 * - Decimal or hex (0x prefix or bare hex is allowed)
 * - # starts a comment
 * - Empty lines are ignored
 * Returns 0 on success, non-zero on error
 */
static int parse_cc_ranges(const char *path, npycrf_cc_range_t **out_ranges, uint32_t *out_count) {
  if (!path || !out_ranges || !out_count) return -1;
  *out_ranges = NULL;
  *out_count = 0;

  FILE *f = fopen(path, "r");
  if (!f) {
    fprintf(stderr, "[cc_ranges] cannot open %s\n", path);
    return -1;
  }

  size_t cap = 64;
  size_t n = 0;
  npycrf_cc_range_t *ranges = (npycrf_cc_range_t *)malloc(cap * sizeof(npycrf_cc_range_t));
  if (!ranges) {
    fclose(f);
    return -2;
  }

  char line[512];
  int line_no = 0;
  while (fgets(line, sizeof(line), f)) {
    line_no++;
    /* skip leading whitespace */
    char *p = line;
    while (*p == ' ' || *p == '\t') p++;
    /* skip empty lines and comments */
    if (*p == '\0' || *p == '\n' || *p == '\r' || *p == '#') continue;

    /* parse: start end class_id */
    char *endptr;
    unsigned long lo = strtoul(p, &endptr, 0);  /* 0 = auto-detect base */
    if (endptr == p) {
      fprintf(stderr, "[cc_ranges] line %d: invalid start value\n", line_no);
      free(ranges);
      fclose(f);
      return -3;
    }
    p = endptr;
    while (*p == ' ' || *p == '\t') p++;

    unsigned long hi = strtoul(p, &endptr, 0);
    if (endptr == p) {
      fprintf(stderr, "[cc_ranges] line %d: invalid end value\n", line_no);
      free(ranges);
      fclose(f);
      return -3;
    }
    p = endptr;
    while (*p == ' ' || *p == '\t') p++;

    unsigned long cid = strtoul(p, &endptr, 0);
    if (endptr == p) {
      fprintf(stderr, "[cc_ranges] line %d: invalid class_id\n", line_no);
      free(ranges);
      fclose(f);
      return -3;
    }

    /* validate */
    if (lo > hi) {
      fprintf(stderr, "[cc_ranges] line %d: start (%lu) > end (%lu)\n", line_no, lo, hi);
      free(ranges);
      fclose(f);
      return -4;
    }
    if (lo > 0x10FFFF || hi > 0x10FFFF) {
      fprintf(stderr, "[cc_ranges] line %d: value out of Unicode range\n", line_no);
      free(ranges);
      fclose(f);
      return -4;
    }
    if (cid > 255) {
      fprintf(stderr, "[cc_ranges] line %d: class_id must be 0-255\n", line_no);
      free(ranges);
      fclose(f);
      return -4;
    }

    /* grow array if needed */
    if (n >= cap) {
      cap *= 2;
      npycrf_cc_range_t *nb = (npycrf_cc_range_t *)realloc(ranges, cap * sizeof(npycrf_cc_range_t));
      if (!nb) {
        free(ranges);
        fclose(f);
        return -2;
      }
      ranges = nb;
    }

    ranges[n].lo = (uint32_t)lo;
    ranges[n].hi = (uint32_t)hi;
    ranges[n].class_id = (uint8_t)cid;
    ranges[n]._pad[0] = 0;
    ranges[n]._pad[1] = 0;
    ranges[n]._pad[2] = 0;
    n++;
  }
  fclose(f);

  if (n == 0) {
    fprintf(stderr, "[cc_ranges] warning: no ranges found in %s\n", path);
    free(ranges);
    return 0;
  }

  /* sort by lo ascending */
  qsort(ranges, n, sizeof(npycrf_cc_range_t), cmp_cc_range_lo);

  /* check for overlaps */
  for (size_t i = 1; i < n; i++) {
    if (ranges[i].lo <= ranges[i-1].hi) {
      fprintf(stderr, "[cc_ranges] overlap detected: [%u-%u] and [%u-%u]\n",
              ranges[i-1].lo, ranges[i-1].hi, ranges[i].lo, ranges[i].hi);
      free(ranges);
      return -5;
    }
  }

  printf("[cc_ranges] loaded %zu ranges from %s\n", n, path);
  *out_ranges = ranges;
  *out_count = (uint32_t)n;
  return 0;
}

/* =====================
 * CLI
 * ===================== */

static void usage(const char *prog) {
  fprintf(stderr,
          "Usage: %s --corpus corpus.txt --out model.bin [options]\n\n"
          "Options:\n"
          "  --vocab N              target vocab size (default: 8000)\n"
          "  --max_piece_len N      max piece length in codepoints (default: 8)\n"
          "  --iters N              EM iterations (default: 5)\n"
          "  --sample_bytes N       bytes used for candidate extraction (default: 20000000)\n"
          "  --cand_total N         total candidates kept (default: 50000)\n"
          "  --min_count N          min ngram count (default: 50)\n"
          "  --char_vocab N         number of single chars kept for UniLM coverage (default: 6000)\n"
          "  --fallback_char C      fallback character for rare chars (UTF-8, default: ?)\n"
          "  --max_line_bytes N     skip lines longer than this (default: 4096)\n"
          "  --max_sentence_cp N    workspace max codepoints per sentence (default: 2048)\n"
          "  --skip_long_cp 0|1     skip sentences longer than max_sentence_cp (default: 1)\n"
          "  --precheck_lines N     precheck coverage on first N sentences (default: 5000, 0=disable)\n"
          "  --keep_single_top N    keep top-N single-char pieces in exported dict (default: 400)\n"
          "  --unk_base X           unknown base penalty (ln, default: -5.0)\n"
          "  --unk_per_cp X         unknown per-cp penalty (ln, default: -1.0)\n"
          "  --lambda0 X            lambda0 for npycrf decode (default: 1.0)\n"
          "  --mdl_lambda0 X        MDL lambda0 (default: 0.0)\n"
          "  --mdl_lambda_len X     MDL lambda_len (default: 0.15)\n"
          "\nCRF options (no hard-coded weights):\n"
          "  --crf_config PATH      override CRF weights from config file\n"
          "  --crf_supervised PATH  train CRF weights from segmented corpus (space-separated tokens)\n"
          "  --crf_epochs N         supervised CRF epochs/iters (default: 20)\n"
          "  --crf_opt sgd|lbfgs     supervised optimizer (default: lbfgs)\n"
          "  --crf_lr X             supervised CRF learning rate (SGD only, default: 0.05)\n"
          "  --crf_l2 X             supervised CRF L2 regularization (default: 1e-4)\n"
          "  --crf_lbfgs_m N         L-BFGS history size (default: 8)\n"
          "  --crf_tol X             L-BFGS gradient-norm tolerance (default: 1e-4)\n"
          "\nUnsupervised CRF training:\n"
          "  --crf_unsupervised 0|1  enable CRF unsupervised training (default: 0)\n"
          "  --crf_unsup_sentences N number of sentences for pseudo-label (default: 1000)\n"
          "\nLossless tokenization:\n"
          "  --lossless_ws 0|1       enable lossless whitespace encoding (default: 0)\n"
          "  --lossless_eol 0|1      append meta-LF to each line for line-based roundtrip (default: 0)\n"
          "\nCharacter class mode:\n"
          "  --cc_mode MODE          character class mode: compat|ascii|utf8len|ranges (default: compat)\n"
          "  --cc_ranges FILE        ranges file for --cc_mode ranges (format: start end class_id per line)\n"
          "  --cc_fallback MODE      fallback mode for ranges: ascii|utf8len (default: utf8len)\n"
          "\n",
          prog);
}

static int arg_eq(const char *a, const char *b) { return strcmp(a, b) == 0; }

int main(int argc, char **argv) {
  const char *corpus_path = NULL;
  const char *out_path = NULL;

  size_t target_vocab = 8000;
  int max_piece_len_cp = 8;
  int iters = 5;
  size_t sample_bytes = 20000000u; /* 20MB */
  size_t cand_total = 50000u;
  uint32_t min_count = 50u;
  size_t max_line_bytes = 4096u;
  size_t max_sentence_cp = 2048u;
  int skip_long_cp = 1;
  size_t precheck_lines = 5000u;
  size_t keep_single_top = 400u;

  size_t char_vocab = 6000u;
  uint32_t fallback_cp = (uint32_t)'?';

  double unk_base = -5.0;
  double unk_per_cp = -1.0;
  double lambda0 = 1.0;

  double mdl_lambda0 = 0.0;
  double mdl_lambda_len = 0.15;

  /* CRF weights (optional)
   *  - --crf_config: override transitions/feature weights without recompiling
   *  - --crf_supervised: learn weights from a small segmented corpus
   */
  const char *crf_config_path = NULL;
  const char *crf_supervised_path = NULL;
  int crf_epochs = 20;
  const char *crf_opt = "lbfgs";
  int crf_lbfgs_m = 8;
  double crf_tol = 1e-4;
  double crf_lr = 0.05;
  double crf_l2 = 1e-4;

  /* unsupervised CRF training */
  int crf_unsupervised = 0;
  size_t crf_unsup_sentences = 1000;

  /* lossless whitespace */
  int lossless_ws = 0;
  int lossless_eol = 0;

  /* character class mode */
  const char *cc_mode_str = "compat";
  const char *cc_ranges_path = NULL;
  const char *cc_fallback_str = "utf8len";

  for (int i = 1; i < argc; i++) {
    if (arg_eq(argv[i], "--corpus") && i + 1 < argc) {
      corpus_path = argv[++i];
    } else if (arg_eq(argv[i], "--out") && i + 1 < argc) {
      out_path = argv[++i];
    } else if (arg_eq(argv[i], "--vocab") && i + 1 < argc) {
      target_vocab = (size_t)strtoul(argv[++i], NULL, 10);
    } else if (arg_eq(argv[i], "--max_piece_len") && i + 1 < argc) {
      max_piece_len_cp = atoi(argv[++i]);
    } else if (arg_eq(argv[i], "--iters") && i + 1 < argc) {
      iters = atoi(argv[++i]);
    } else if (arg_eq(argv[i], "--sample_bytes") && i + 1 < argc) {
      sample_bytes = (size_t)strtoull(argv[++i], NULL, 10);
    } else if (arg_eq(argv[i], "--cand_total") && i + 1 < argc) {
      cand_total = (size_t)strtoull(argv[++i], NULL, 10);
    } else if (arg_eq(argv[i], "--min_count") && i + 1 < argc) {
      min_count = (uint32_t)strtoul(argv[++i], NULL, 10);
    } else if (arg_eq(argv[i], "--char_vocab") && i + 1 < argc) {
      char_vocab = (size_t)strtoull(argv[++i], NULL, 10);
    } else if (arg_eq(argv[i], "--fallback_char") && i + 1 < argc) {
      /* UTF-8 1文字だけを想定 */
      const char *fc = argv[++i];
      uint32_t cp = 0;
      size_t adv = 0;
      if (utf8_decode1((const uint8_t *)fc, strlen(fc), 0, &cp, &adv) && adv > 0) {
        fallback_cp = cp;
      } else {
        fallback_cp = (uint32_t)'?';
      }
    } else if (arg_eq(argv[i], "--max_line_bytes") && i + 1 < argc) {
      max_line_bytes = (size_t)strtoull(argv[++i], NULL, 10);
    } else if (arg_eq(argv[i], "--max_sentence_cp") && i + 1 < argc) {
      max_sentence_cp = (size_t)strtoull(argv[++i], NULL, 10);
    } else if (arg_eq(argv[i], "--skip_long_cp") && i + 1 < argc) {
      skip_long_cp = atoi(argv[++i]) ? 1 : 0;
    } else if (arg_eq(argv[i], "--precheck_lines") && i + 1 < argc) {
      precheck_lines = (size_t)strtoull(argv[++i], NULL, 10);
    } else if (arg_eq(argv[i], "--keep_single_top") && i + 1 < argc) {
      keep_single_top = (size_t)strtoull(argv[++i], NULL, 10);
    } else if (arg_eq(argv[i], "--unk_base") && i + 1 < argc) {
      unk_base = atof(argv[++i]);
    } else if (arg_eq(argv[i], "--unk_per_cp") && i + 1 < argc) {
      unk_per_cp = atof(argv[++i]);
    } else if (arg_eq(argv[i], "--lambda0") && i + 1 < argc) {
      lambda0 = atof(argv[++i]);
    } else if (arg_eq(argv[i], "--mdl_lambda0") && i + 1 < argc) {
      mdl_lambda0 = atof(argv[++i]);
    } else if (arg_eq(argv[i], "--mdl_lambda_len") && i + 1 < argc) {
      mdl_lambda_len = atof(argv[++i]);
    } else if (arg_eq(argv[i], "--crf_config") && i + 1 < argc) {
      crf_config_path = argv[++i];
    } else if (arg_eq(argv[i], "--crf_supervised") && i + 1 < argc) {
      crf_supervised_path = argv[++i];
    } else if (arg_eq(argv[i], "--crf_epochs") && i + 1 < argc) {
      crf_epochs = atoi(argv[++i]);
    } else if (arg_eq(argv[i], "--crf_opt") && i + 1 < argc) {
      crf_opt = argv[++i];
    } else if (arg_eq(argv[i], "--crf_lbfgs_m") && i + 1 < argc) {
      crf_lbfgs_m = atoi(argv[++i]);
    } else if (arg_eq(argv[i], "--crf_tol") && i + 1 < argc) {
      crf_tol = strtod(argv[++i], NULL);
    } else if (arg_eq(argv[i], "--crf_lr") && i + 1 < argc) {
      crf_lr = atof(argv[++i]);
    } else if (arg_eq(argv[i], "--crf_l2") && i + 1 < argc) {
      crf_l2 = atof(argv[++i]);
    } else if (arg_eq(argv[i], "--crf_unsupervised") && i + 1 < argc) {
      crf_unsupervised = atoi(argv[++i]);
    } else if (arg_eq(argv[i], "--crf_unsup_sentences") && i + 1 < argc) {
      crf_unsup_sentences = (size_t)atol(argv[++i]);
    } else if (arg_eq(argv[i], "--lossless_ws") && i + 1 < argc) {
      lossless_ws = atoi(argv[++i]);
    } else if (arg_eq(argv[i], "--lossless_eol") && i + 1 < argc) {
      lossless_eol = atoi(argv[++i]);
    } else if (arg_eq(argv[i], "--cc_mode") && i + 1 < argc) {
      cc_mode_str = argv[++i];
    } else if (arg_eq(argv[i], "--cc_ranges") && i + 1 < argc) {
      cc_ranges_path = argv[++i];
    } else if (arg_eq(argv[i], "--cc_fallback") && i + 1 < argc) {
      cc_fallback_str = argv[++i];
    } else if (arg_eq(argv[i], "--help") || arg_eq(argv[i], "-h")) {
      usage(argv[0]);
      return 0;
    } else {
      fprintf(stderr, "unknown arg: %s\n", argv[i]);
      usage(argv[0]);
      return 2;
    }
  }

  if (!corpus_path || !out_path) {
    usage(argv[0]);
    return 2;
  }

  printf("[mmjp_train] corpus=%s\n", corpus_path);
  printf("[mmjp_train] target_vocab=%zu max_piece_len_cp=%d iters=%d\n", target_vocab, max_piece_len_cp, iters);
  printf("[mmjp_train] limits: max_line_bytes=%zu max_sentence_cp=%zu skip_long_cp=%d\n", max_line_bytes, max_sentence_cp, skip_long_cp);
  if (lossless_ws || lossless_eol) {
    printf("[mmjp_train] lossless_ws=%d lossless_eol=%d\n", lossless_ws, lossless_eol);
  }

  /* --- pass 1: count codepoints (for coverage character set) --- */
  FILE *fc = fopen(corpus_path, "rb");
  if (!fc) {
    fprintf(stderr, "failed to open corpus\n");
    return 1;
  }
  u32cnt_t cpmap;
  if (!u32cnt_init(&cpmap, 1u << 16)) {
    fprintf(stderr, "oom\n");
    fclose(fc);
    return 1;
  }

  file_iter_t fit;
  memset(&fit, 0, sizeof(fit));
  fit.f = fc;
  fit.max_line_bytes = max_line_bytes;
  fit.max_sentence_cp = max_sentence_cp;
  fit.skip_long_cp = skip_long_cp;
  fit.keep_chars = NULL; /* no mapping in counting pass */
  fit.fallback_cp = fallback_cp;
  fit.append_eol = lossless_eol;

  size_t n_lines = 0;
  while (1) {
    int r = file_iter_readline(&fit);
    if (r < 0) {
      fprintf(stderr, "read error\n");
      u32cnt_free(&cpmap);
      fclose(fc);
      free(fit.buf);
      free(fit.mapped);
      return 1;
    }
    if (r == 0) break;
    if (fit.len == 0) continue;
    n_lines++;
    const uint8_t *s = fit.buf;
    size_t len = fit.len;
    for (size_t pos = 0; pos < len; ) {
      uint32_t cp = 0;
      size_t adv = 0;
      if (!utf8_decode1(s, len, pos, &cp, &adv) || adv == 0) {
        /* 不正バイト列は学習側では raw byte として扱わず、フォールバックに畳み込む */
        cp = fallback_cp;
        adv = 1;
      }
      pos += adv;
      (void)u32cnt_inc(&cpmap, cp);
    }
  }

  printf("[mmjp_train] scanned %zu lines, unique codepoints=%zu\n", n_lines, cpmap.size);

  /* --- build keep_chars set --- */
  if (target_vocab > 0 && char_vocab >= target_vocab) {
    /* mandatory single chars が target_vocab を食い尽くさないように */
    char_vocab = (target_vocab >= 512) ? (target_vocab / 2) : (target_vocab - 1);
  }
  if (char_vocab < 256) char_vocab = 256;

  u32set_t keep_chars;
  if (!u32set_init(&keep_chars, 1u << 14)) {
    fprintf(stderr, "oom keep_chars\n");
    u32cnt_free(&cpmap);
    fclose(fc);
    free(fit.buf);
    free(fit.mapped);
    return 1;
  }
  (void)u32set_insert(&keep_chars, fallback_cp);
  /* ASCII printable は常に保持（ログ/デバッグ/数値などで役立つ） */
  for (uint32_t cp = 0x20; cp <= 0x7E; cp++) (void)u32set_insert(&keep_chars, cp);

  cpair_t *arr = (cpair_t *)malloc(cpmap.size * sizeof(cpair_t));
  if (!arr) {
    fprintf(stderr, "oom arr\n");
    u32set_free(&keep_chars);
    u32cnt_free(&cpmap);
    fclose(fc);
    free(fit.buf);
    free(fit.mapped);
    return 1;
  }
  size_t arr_n = 0;
  for (size_t i = 0; i < cpmap.cap; i++) {
    if (cpmap.cp[i] == 0) continue;
    uint32_t cp = cpmap.cp[i];
    /* 改行/CR/タブは分かち書き対象にしない */
    if (cp == '\n' || cp == '\r' || cp == '\t') continue;
    arr[arr_n].cp = cp;
    arr[arr_n].cnt = cpmap.cnt[i];
    arr_n++;
  }
  qsort(arr, arr_n, sizeof(cpair_t), cmp_cpair_desc);

  size_t added = keep_chars.size;
  for (size_t i = 0; i < arr_n && added < char_vocab; i++) {
    if (u32set_contains(&keep_chars, arr[i].cp)) continue;
    if (u32set_insert(&keep_chars, arr[i].cp)) added++;
  }
  free(arr);
  u32cnt_free(&cpmap);

  printf("[mmjp_train] keep_chars=%zu (char_vocab=%zu, fallback=%u)\n", keep_chars.size, char_vocab, fallback_cp);

  /* enable mapping for subsequent passes */
  fit.keep_chars = &keep_chars;
  fit.fallback_cp = fallback_cp;
  fit.stat_skipped_long_bytes = 0;
  fit.stat_skipped_long_cp = 0;

  /* --- candidate extraction (mapped sample) --- */
  file_iter_reset(&fit);
  uint8_t *sample = (uint8_t *)malloc(sample_bytes + 1024u);
  if (!sample) {
    fprintf(stderr, "oom sample\n");
    u32set_free(&keep_chars);
    fclose(fc);
    free(fit.buf);
    free(fit.mapped);
    return 1;
  }
  size_t sample_cap = sample_bytes + 1024u;
  size_t sample_len = 0;
  while (sample_len + 1 < sample_bytes) {
    int r = file_iter_readline(&fit);
    if (r < 0) {
      fprintf(stderr, "read error during sample\n");
      free(sample);
      u32set_free(&keep_chars);
      fclose(fc);
      free(fit.buf);
      free(fit.mapped);
      return 1;
    }
    if (r == 0) break;
    if (fit.len == 0) continue;
    const uint8_t *src = (fit.mapped_len > 0) ? fit.mapped : fit.buf;
    size_t slen = (fit.mapped_len > 0) ? fit.mapped_len : fit.len;
    if (sample_len + slen + 2 > sample_cap) {
      size_t nc = sample_cap * 2;
      while (nc < sample_len + slen + 2) nc *= 2;
      uint8_t *nb = (uint8_t *)realloc(sample, nc);
      if (!nb) {
        fprintf(stderr, "oom sample grow\n");
        free(sample);
        u32set_free(&keep_chars);
        fclose(fc);
        free(fit.buf);
        free(fit.mapped);
        return 1;
      }
      sample = nb;
      sample_cap = nc;
    }
    memcpy(&sample[sample_len], src, slen);
    sample_len += slen;
    sample[sample_len++] = '\n';
    if (sample_len >= sample_bytes) break;
  }
  sample[sample_len] = 0;

  printf("[mmjp_train] candidate sample bytes=%zu (mapped)\n", sample_len);

  cand_t *cands = NULL;
  size_t cands_n = 0;
  {
    uint8_t fb[4];
    size_t fb_len = utf8_encode1(fallback_cp, fb);
    if (!collect_top_ngrams(sample, sample_len, max_piece_len_cp, cand_total, min_count, fb, fb_len, &cands, &cands_n)) {
      fprintf(stderr, "candidate extraction failed\n");
      free(sample);
      u32set_free(&keep_chars);
      fclose(fc);
      free(fit.buf);
      free(fit.mapped);
      return 1;
    }
  }
  printf("[mmjp_train] candidates=%zu\n", cands_n);

  free(sample);

  /* --- init unilm model --- */
  size_t mandatory_count = keep_chars.size;
  size_t vocab_cap = mandatory_count + cands_n + 16;

  size_t str_cap = 1024;
  for (size_t i = 0; i < keep_chars.cap; i++) {
    if (keep_chars.k[i] == 0) continue;
    uint8_t tmp[4];
    str_cap += utf8_encode1(keep_chars.k[i], tmp);
  }
  for (size_t i = 0; i < cands_n; i++) str_cap += cands[i].len_bytes;
  str_cap += 1024;

  size_t da_cap = 256;
  while (da_cap < str_cap * 2 + 512) da_cap <<= 1;

  unilm_model_t um;
  if (unilm_model_init_dynamic(&um, vocab_cap, str_cap, da_cap) != UNILM_OK) {
    fprintf(stderr, "unilm_model_init_dynamic failed\n");
    for (size_t i = 0; i < cands_n; i++) cand_free(&cands[i]);
    free(cands);
    u32set_free(&keep_chars);
    fclose(fc);
    free(fit.buf);
    free(fit.mapped);
    return 1;
  }

  /* add mandatory single codepoints */
  size_t added_single = 0;
  for (size_t i = 0; i < keep_chars.cap; i++) {
    uint32_t cp = keep_chars.k[i];
    if (cp == 0) continue;
    uint8_t tmp[4];
    size_t blen = utf8_encode1(cp, tmp);
    if (unilm_model_add_piece(&um, tmp, blen, UNILM_PIECE_MANDATORY) < 0) {
      fprintf(stderr, "add piece failed (single)\n");
      /* continue */
    } else {
      added_single++;
    }
  }
  printf("[mmjp_train] mandatory singles added=%zu\n", added_single);

  /* add candidates */
  size_t added_cand = 0;
  for (size_t i = 0; i < cands_n; i++) {
    if (unilm_model_add_piece(&um, (const uint8_t *)cands[i].s, cands[i].len_bytes, 0) < 0) {
      /* ignore */
    } else {
      added_cand++;
    }
  }
  printf("[mmjp_train] candidates added=%zu (requested=%zu)\n", added_cand, cands_n);

  for (size_t i = 0; i < cands_n; i++) cand_free(&cands[i]);
  free(cands);
  /* keep_chars は学習コーパスのマッピングに使うので、ここでは free しない */

  /*
   * 重要:
   *  - keep_chars はハッシュ順で追加されるため、トライ挿入順がランダムになりがち。
   *  - 大きい語彙では挿入順による衝突/再配置が増え、まれに NOCOVER の原因になる。
   *  - ここで一度、語彙を辞書順でトライに積み直して安定化する。
   */
  {
    int rc = unilm_model_rebuild_trie_sorted(&um);
    if (rc != UNILM_OK) {
      fprintf(stderr, "unilm_model_rebuild_trie_sorted failed rc=%d\n", rc);
      unilm_model_free(&um);
      u32set_free(&keep_chars);
      fclose(fc);
      free(fit.buf);
      free(fit.mapped);
      return 1;
    }
  }

  /* --- UniLM training (EM+MDL) --- */
  unilm_train_config_t cfg;
  memset(&cfg, 0, sizeof(cfg));
  cfg.num_iters = iters;
  cfg.max_piece_len_cp = max_piece_len_cp;
  cfg.smoothing = (unilm_real_t)0.1;
  cfg.mdl_lambda0 = (unilm_real_t)mdl_lambda0;
  cfg.mdl_lambda_len = (unilm_real_t)mdl_lambda_len;
  cfg.target_vocab_size = target_vocab;
  cfg.prune_each_iter = 1;
  cfg.min_prob = (unilm_real_t)1e-12;

  unilm_workspace_t wk;
  memset(&wk, 0, sizeof(wk));
  size_t heap_cap = (cfg.target_vocab_size > 0) ? cfg.target_vocab_size : um.vocab_size;
  if (unilm_workspace_init_dynamic(&wk, max_sentence_cp, um.vocab_size, heap_cap) != UNILM_OK) {
    fprintf(stderr, "workspace init failed\n");
    unilm_model_free(&um);
    fclose(fc);
    free(fit.buf);
    return 1;
  }

  unilm_real_t *counts = (unilm_real_t *)calloc(um.vocab_size, sizeof(unilm_real_t));
  if (!counts) {
    fprintf(stderr, "oom counts\n");
    unilm_workspace_free(&wk);
    unilm_model_free(&um);
    fclose(fc);
    free(fit.buf);
    return 1;
  }

  /* corpus iterator */
  unilm_corpus_iter_t it;
  it.next = file_corpus_next;
  it.reset = file_corpus_reset;
  it.user = &fit;

  /* initialize logp uniform */
  {
    unilm_real_t lp = (unilm_real_t)(-log((double)um.vocab_size));
    for (size_t i = 0; i < um.vocab_size; i++) um.logp[i] = lp;
    (void)unilm_model_normalize(&um, cfg.min_prob);
  }

  /* 事前にカバレッジ（NOCOVER）を軽くチェックして、落ちる場合は原因を表示 */
  if (precheck_lines > 0) {
    printf("[mmjp_train] precheck coverage (first %zu sentences)\n", precheck_lines);
    fit.stat_skipped_long_bytes = 0;
    fit.stat_skipped_long_cp = 0;
    size_t out_cap = max_sentence_cp + 8u;
    int prc = mmjp_locate_first_nocover(&um, &fit, &wk, max_piece_len_cp, out_cap, precheck_lines);
    if (prc != 0) {
      fprintf(stderr, "precheck failed (NOCOVER or error) rc=%d\n", prc);
      free(counts);
      unilm_workspace_free(&wk);
      unilm_model_free(&um);
      u32set_free(&keep_chars);
      fclose(fc);
      free(fit.buf);
      free(fit.mapped);
      return 1;
    }
    if (it.reset) it.reset(it.user);
  }

  printf("[mmjp_train] EM+MDL start (vocab=%zu)\n", um.vocab_size);
  for (int iter = 0; iter < cfg.num_iters; iter++) {
    if (it.reset) it.reset(it.user);
    unilm_em_stats_t st;
    memset(&st, 0, sizeof(st));
    fit.stat_skipped_long_bytes = 0;
    fit.stat_skipped_long_cp = 0;
    int rc2 = unilm_em_e_step(&um, &it, &cfg, &wk, counts, &st);
    if (rc2 != UNILM_OK) {
      fprintf(stderr, "E-step failed rc=%d\n", rc2);
      if (rc2 == UNILM_ERR_NOCOVER) {
        size_t out_cap = max_sentence_cp + 8u;
        (void)mmjp_locate_first_nocover(&um, &fit, &wk, max_piece_len_cp, out_cap, 0);
      }
      free(counts);
      unilm_workspace_free(&wk);
      unilm_model_free(&um);
      u32set_free(&keep_chars);
      fclose(fc);
      free(fit.buf);
      free(fit.mapped);
      return 1;
    }
    rc2 = unilm_em_m_step(&um, &cfg, counts);
    if (rc2 != UNILM_OK) {
      fprintf(stderr, "M-step failed rc=%d\n", rc2);
      free(counts);
      unilm_workspace_free(&wk);
      unilm_model_free(&um);
      fclose(fc);
      free(fit.buf);
      return 1;
    }
    int newV = unilm_prune_mdl(&um, &cfg, &wk, counts);
    if (newV < 0) {
      fprintf(stderr, "prune failed rc=%d\n", newV);
      free(counts);
      unilm_workspace_free(&wk);
      unilm_model_free(&um);
      fclose(fc);
      free(fit.buf);
      return 1;
    }
    printf("  iter %d: loglik=%.3f n_sent=%.0f n_tok_exp=%.1f vocab=%d (skipped_bytes=%zu skipped_cp=%zu)\n",
           iter + 1, (double)st.loglik, (double)st.n_sent, (double)st.n_tokens_exp, newV,
           fit.stat_skipped_long_bytes, fit.stat_skipped_long_cp);
  }

  printf("[mmjp_train] UniLM done. vocab=%zu\n", um.vocab_size);

  /* --- export selection --- */
  /* keep all multi-char, plus selected singles */
  uint8_t *keep = (uint8_t *)calloc(um.vocab_size, 1);
  if (!keep) {
    fprintf(stderr, "oom keep\n");
    free(counts);
    unilm_workspace_free(&wk);
    unilm_model_free(&um);
    fclose(fc);
    free(fit.buf);
    return 1;
  }

  /* always keep multi-char */
  size_t multi_keep = 0;
  size_t single_total = 0;
  idscore_t *single = (idscore_t *)malloc(um.vocab_size * sizeof(idscore_t));
  if (!single) {
    fprintf(stderr, "oom single\n");
    free(keep);
    free(counts);
    unilm_workspace_free(&wk);
    unilm_model_free(&um);
    fclose(fc);
    free(fit.buf);
    return 1;
  }

  for (uint32_t id = 0; id < (uint32_t)um.vocab_size; id++) {
    unilm_piece_t *p = &um.pieces[id];
    if (p->len_cp >= 2) {
      keep[id] = 1;
      multi_keep++;
    } else {
      /* single */
      double prob = exp((double)um.logp[id]);
      single[single_total].id = id;
      single[single_total].p = prob;
      single_total++;
    }
  }

  qsort(single, single_total, sizeof(idscore_t), cmp_idscore_desc);
  size_t keep_singles = 0;
  for (size_t i = 0; i < single_total && keep_singles < keep_single_top; i++) {
    uint32_t id = single[i].id;
    keep[id] = 1;
    keep_singles++;
  }

  free(single);
  printf("[mmjp_train] export keep: multi=%zu singles_top=%zu -> total_keep=~%zu\n", multi_keep, keep_singles, multi_keep + keep_singles);

  /* build npycrf trie and logp_uni */
  da_trie_t da;
  if (da_trie_init_dynamic(&da, 1024) != DA_OK) {
    fprintf(stderr, "da init failed\n");
    free(keep);
    free(counts);
    unilm_workspace_free(&wk);
    unilm_model_free(&um);
    fclose(fc);
    free(fit.buf);
    return 1;
  }

  /* map unilm id -> new id */
  uint16_t *map = (uint16_t *)malloc(um.vocab_size * sizeof(uint16_t));
  if (!map) {
    fprintf(stderr, "oom map\n");
    da_trie_free(&da);
    free(keep);
    free(counts);
    unilm_workspace_free(&wk);
    unilm_model_free(&um);
    fclose(fc);
    free(fit.buf);
    return 1;
  }
  for (size_t i = 0; i < um.vocab_size; i++) map[i] = 0xFFFFu;

  size_t export_vocab = 0;
  for (uint32_t id = 0; id < (uint32_t)um.vocab_size; id++) {
    if (!keep[id]) continue;
    map[id] = (uint16_t)export_vocab;
    export_vocab++;
  }
  if (export_vocab > 0xFFFEu) {
    fprintf(stderr, "export vocab too large\n");
    free(map);
    da_trie_free(&da);
    free(keep);
    free(counts);
    unilm_workspace_free(&wk);
    unilm_model_free(&um);
    fclose(fc);
    free(fit.buf);
    return 1;
  }

  int16_t *logp_uni = (int16_t *)malloc(export_vocab * sizeof(int16_t));
  if (!logp_uni) {
    fprintf(stderr, "oom logp_uni\n");
    free(map);
    da_trie_free(&da);
    free(keep);
    free(counts);
    unilm_workspace_free(&wk);
    unilm_model_free(&um);
    fclose(fc);
    free(fit.buf);
    return 1;
  }

  /* insert pieces into trie */
  for (uint32_t id = 0; id < (uint32_t)um.vocab_size; id++) {
    if (!keep[id]) continue;
    size_t blen = 0;
    const uint8_t *b = unilm_model_piece_bytes(&um, id, &blen);
    if (!b || blen == 0) continue;
    uint16_t nid = map[id];
    (void)npycrf_da_set_term_value(&da, b, blen, nid);
    logp_uni[nid] = q88_from_double((double)um.logp[id]);
  }

  /* finalize npycrf model struct */
  crf_table_t crf;
  if (!crf_table_build_ja_basic(&crf)) {
    fprintf(stderr, "crf preset build failed\n");
    free(logp_uni);
    free(map);
    da_trie_free(&da);
    free(keep);
    free(counts);
    unilm_workspace_free(&wk);
    unilm_model_free(&um);
    fclose(fc);
    free(fit.buf);
    return 1;
  }

  /* --- CRF weights (defaults -> optional config -> optional supervised) --- */
  double trans00 = 0.2;
  double trans01 = -0.4;
  double trans10 = 0.0;
  double trans11 = -0.6;
  double bos_to1 = 0.5;
  double *feat_w_d = (double *)malloc((size_t)crf.n * sizeof(double));
  if (!feat_w_d) {
    fprintf(stderr, "oom (feat_w_d)\n");
    crf_table_free(&crf);
    free(logp_uni);
    free(map);
    da_trie_free(&da);
    free(keep);
    free(counts);
    unilm_workspace_free(&wk);
    unilm_model_free(&um);
    fclose(fc);
    free(fit.buf);
    return 1;
  }
  for (uint16_t i = 0; i < crf.n; i++) feat_w_d[i] = q88_to_double(crf.w[i]);

  if (crf_config_path) {
    printf("[mmjp_train] CRF config: %s\n", crf_config_path);
    (void)crf_apply_config_file(crf_config_path, &trans00, &trans01, &trans10, &trans11, &bos_to1, &crf, feat_w_d);
  }

  if (crf_supervised_path) {
    printf("[mmjp_train] CRF supervised: %s\n", crf_supervised_path);
    crf_dataset_t ds;
    if (!crf_dataset_load(crf_supervised_path, max_line_bytes, max_sentence_cp, &ds) || ds.n == 0) {
      fprintf(stderr, "[mmjp_train] CRF supervised: no usable sentences\n");
    } else {
      printf("[mmjp_train] CRF supervised: sentences=%zu total_pos=%zu\n", ds.n, ds.total_pos);
            if (crf_opt && strcmp(crf_opt, "sgd") == 0) {
        (void)crf_train_supervised(&ds, &crf, feat_w_d, &trans00, &trans01, &trans10, &trans11, crf_epochs, crf_lr, crf_l2);
      } else {
        (void)crf_train_supervised_lbfgs(&ds, &crf, feat_w_d, &trans00, &trans01, &trans10, &trans11, crf_epochs, crf_l2, crf_lbfgs_m, crf_tol);
      }
    }
    crf_dataset_free(&ds);
  }

  /* Unsupervised CRF training (pseudo-labels from LM Viterbi) */
  if (crf_unsupervised) {
    printf("[mmjp_train] CRF unsupervised: pseudo-label = LM-only (CRF disabled)\n");
    printf("[mmjp_train] CRF unsupervised: lambda0=%.4f (for final model, not used in pseudo-label generation)\n", lambda0);
    printf("[mmjp_train] CRF unsupervised: generating pseudo-labels...\n");
    crf_dataset_t ds;
    if (!crf_dataset_from_lm_viterbi(corpus_path, max_line_bytes, max_sentence_cp,
                                     &um, &wk, max_piece_len_cp,
                                     crf_unsup_sentences, &ds) || ds.n == 0) {
      fprintf(stderr, "[mmjp_train] CRF unsupervised: no usable sentences\n");
    } else {
      printf("[mmjp_train] CRF unsupervised: sentences=%zu total_pos=%zu\n", ds.n, ds.total_pos);
      if (crf_opt && strcmp(crf_opt, "sgd") == 0) {
        (void)crf_train_supervised(&ds, &crf, feat_w_d, &trans00, &trans01, &trans10, &trans11, crf_epochs, crf_lr, crf_l2);
      } else {
        (void)crf_train_supervised_lbfgs(&ds, &crf, feat_w_d, &trans00, &trans01, &trans10, &trans11, crf_epochs, crf_l2, crf_lbfgs_m, crf_tol);
      }
    }
    crf_dataset_free(&ds);
  }

  /* quantize back to Q8.8 */
  for (uint16_t i = 0; i < crf.n; i++) crf.w[i] = q88_from_double(feat_w_d[i]);
  free(feat_w_d);

  npycrf_model_t nm;
  memset(&nm, 0, sizeof(nm));
  nm.max_word_len = (uint16_t)max_piece_len_cp;

  nm.lm.trie.base = da.base;
  nm.lm.trie.check = da.check;
  nm.lm.trie.capacity = da.capacity;
  nm.lm.logp_uni = logp_uni;
  nm.lm.vocab_size = (uint32_t)export_vocab;
  nm.lm.bigram_key = NULL;
  nm.lm.logp_bi = NULL;
  nm.lm.bigram_size = 0;
  nm.lm.unk_base = q88_from_double(unk_base);
  nm.lm.unk_per_cp = q88_from_double(unk_per_cp);
  nm.lambda0 = q88_from_double(lambda0);

  nm.crf.trans00 = q88_from_double(trans00);
  nm.crf.trans01 = q88_from_double(trans01);
  nm.crf.trans10 = q88_from_double(trans10);
  nm.crf.trans11 = q88_from_double(trans11);
  nm.crf.bos_to1 = q88_from_double(bos_to1);
  nm.crf.feat_key = crf.k;
  nm.crf.feat_w = crf.w;
  nm.crf.feat_count = crf.n;

  /* Set model flags */
  if (lossless_ws) {
    nm.flags |= NPYCRF_FLAG_LOSSLESS_WS;
  }

  /* cc settings */
  npycrf_cc_range_t *cc_ranges = NULL;
  uint32_t cc_range_count = 0;
  npycrf_cc_mode_t cc_mode = NPYCRF_CC_MODE_COMPAT;
  npycrf_cc_mode_t cc_fallback = NPYCRF_CC_MODE_UTF8LEN;

  /* parse cc_mode string */
  if (strcmp(cc_mode_str, "compat") == 0) {
    cc_mode = NPYCRF_CC_MODE_COMPAT;
  } else if (strcmp(cc_mode_str, "ascii") == 0) {
    cc_mode = NPYCRF_CC_MODE_ASCII;
  } else if (strcmp(cc_mode_str, "utf8len") == 0) {
    cc_mode = NPYCRF_CC_MODE_UTF8LEN;
  } else if (strcmp(cc_mode_str, "ranges") == 0) {
    cc_mode = NPYCRF_CC_MODE_RANGES;
  } else {
    fprintf(stderr, "[mmjp_train] unknown cc_mode: %s (expected: compat|ascii|utf8len|ranges)\n", cc_mode_str);
    return 1;
  }

  /* parse cc_fallback string */
  if (strcmp(cc_fallback_str, "ascii") == 0) {
    cc_fallback = NPYCRF_CC_MODE_ASCII;
  } else if (strcmp(cc_fallback_str, "utf8len") == 0) {
    cc_fallback = NPYCRF_CC_MODE_UTF8LEN;
  } else {
    fprintf(stderr, "[mmjp_train] unknown cc_fallback: %s (expected: ascii|utf8len)\n", cc_fallback_str);
    return 1;
  }

  /* load cc_ranges file if needed */
  if (cc_mode == NPYCRF_CC_MODE_RANGES) {
    if (!cc_ranges_path) {
      fprintf(stderr, "[mmjp_train] --cc_mode ranges requires --cc_ranges FILE\n");
      return 1;
    }
    if (parse_cc_ranges(cc_ranges_path, &cc_ranges, &cc_range_count) != 0) {
      return 1;
    }
  }

  nm.cc.mode = cc_mode;
  nm.cc.fallback = cc_fallback;
  nm.cc.ranges = cc_ranges;
  nm.cc.range_count = cc_range_count;

  /* Set cc mode flags */
  if (cc_mode == NPYCRF_CC_MODE_ASCII) {
    nm.flags |= NPYCRF_FLAG_CC_ASCII;
  } else if (cc_mode == NPYCRF_CC_MODE_UTF8LEN) {
    nm.flags |= NPYCRF_FLAG_CC_UTF8LEN;
  } else if (cc_mode == NPYCRF_CC_MODE_RANGES) {
    nm.flags |= NPYCRF_FLAG_CC_RANGES;
  } else if (cc_mode == NPYCRF_CC_MODE_COMPAT) {
    nm.flags |= NPYCRF_FLAG_CC_COMPAT;
  }

  if (cc_mode != NPYCRF_CC_MODE_COMPAT) {
    printf("[mmjp_train] cc_mode=%s cc_fallback=%s cc_range_count=%u\n",
           cc_mode_str, cc_fallback_str, cc_range_count);
  }

  /* --- save model --- */
  printf("[mmjp_train] saving model: vocab=%zu da_cap=%zu feat=%u -> %s\n", export_vocab, da.capacity, crf.n, out_path);
  int s_rc = mmjp_model_save_bin(out_path, &nm);
  if (s_rc != 0) {
    fprintf(stderr, "save failed rc=%d\n", s_rc);
    /* continue to free */
  } else {
    printf("[mmjp_train] done.\n");
  }

  /* --- cleanup --- */
  crf_table_free(&crf);
  free(logp_uni);
  free(map);
  da_trie_free(&da);
  free(keep);
  free(counts);
  unilm_workspace_free(&wk);
  unilm_model_free(&um);
  fclose(fc);
  free(fit.buf);
  free(fit.mapped);
  u32set_free(&keep_chars);
  free(cc_ranges);
  return (s_rc == 0) ? 0 : 1;
}
