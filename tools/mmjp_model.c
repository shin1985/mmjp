#include "mmjp_model.h"

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

/* =====================
 * 小さなLEエンコード/デコード
 * ===================== */

static void wr_u32(FILE *f, uint32_t v) {
  uint8_t b[4] = {
      (uint8_t)(v & 0xFFu),
      (uint8_t)((v >> 8) & 0xFFu),
      (uint8_t)((v >> 16) & 0xFFu),
      (uint8_t)((v >> 24) & 0xFFu),
  };
  fwrite(b, 1, 4, f);
}

static void wr_i16(FILE *f, int16_t v) {
  uint16_t u = (uint16_t)v;
  uint8_t b[2] = {(uint8_t)(u & 0xFFu), (uint8_t)((u >> 8) & 0xFFu)};
  fwrite(b, 1, 2, f);
}

static int rd_u32(FILE *f, uint32_t *out) {
  uint8_t b[4];
  if (fread(b, 1, 4, f) != 4) return 0;
  *out = (uint32_t)b[0] | ((uint32_t)b[1] << 8) | ((uint32_t)b[2] << 16) | ((uint32_t)b[3] << 24);
  return 1;
}

static int rd_i16(FILE *f, int16_t *out) {
  uint8_t b[2];
  if (fread(b, 1, 2, f) != 2) return 0;
  uint16_t u = (uint16_t)b[0] | ((uint16_t)b[1] << 8);
  *out = (int16_t)u;
  return 1;
}

/* =====================
 * バイナリフォーマット
 * ===================== */

/*
 * ヘッダ（可変長の配列本体はこの後に続く）
 *
 * すべて little-endian。
 * base/check は da_index_t のサイズに合わせるのが理想だが、
 * ここでは CLI 前提で int32 を固定で保存する（MCUはCヘッダ埋め込み推奨）。
 */

enum {
  MMJP_DA_INDEX_BYTES = 4u,
};

int mmjp_model_save_bin(const char *path, const npycrf_model_t *m) {
  if (!path || !m) return -1;
  if (!m->lm.trie.base || !m->lm.trie.check || m->lm.trie.capacity == 0) return -2;
  if (!m->lm.logp_uni || m->lm.vocab_size == 0) return -3;

  FILE *f = fopen(path, "wb");
  if (!f) return -10;

  /* --- header (v2) --- */
  fwrite(MMJP_MODEL_MAGIC, 1, 8, f);
  wr_u32(f, MMJP_MODEL_VERSION);
  wr_u32(f, MMJP_DA_INDEX_BYTES);
  wr_u32(f, (uint32_t)m->lm.trie.capacity);
  wr_u32(f, (uint32_t)m->lm.vocab_size);
  wr_u32(f, (uint32_t)m->max_word_len);

  /* unknown/lambda */
  wr_i16(f, m->lm.unk_base);
  wr_i16(f, m->lm.unk_per_cp);
  wr_i16(f, m->lambda0);

  /* CRF transitions */
  wr_i16(f, m->crf.trans00);
  wr_i16(f, m->crf.trans01);
  wr_i16(f, m->crf.trans10);
  wr_i16(f, m->crf.trans11);
  wr_i16(f, m->crf.bos_to1);

  wr_u32(f, m->crf.feat_count);
  wr_u32(f, m->lm.bigram_size);

  /* v2: flags, cc_mode, cc_fallback, cc_range_count */
  wr_u32(f, m->flags);
  uint8_t cc_mode = (uint8_t)m->cc.mode;
  uint8_t cc_fallback = (uint8_t)m->cc.fallback;
  fwrite(&cc_mode, 1, 1, f);
  fwrite(&cc_fallback, 1, 1, f);
  /* 2バイトパディング */
  uint8_t pad2[2] = {0, 0};
  fwrite(pad2, 1, 2, f);
  wr_u32(f, m->cc.range_count);

  /* --- arrays --- */
  /* base/check: int32_t として保存 */
  for (size_t i = 0; i < m->lm.trie.capacity; i++) {
    int32_t v = (int32_t)m->lm.trie.base[i];
    wr_u32(f, (uint32_t)v);
  }
  for (size_t i = 0; i < m->lm.trie.capacity; i++) {
    int32_t v = (int32_t)m->lm.trie.check[i];
    wr_u32(f, (uint32_t)v);
  }

  /* unigram logp Q8.8 */
  fwrite(m->lm.logp_uni, sizeof(int16_t), m->lm.vocab_size, f);

  /* bigram (optional) */
  if (m->lm.bigram_size > 0) {
    if (!m->lm.bigram_key || !m->lm.logp_bi) {
      fclose(f);
      return -4;
    }
    fwrite(m->lm.bigram_key, sizeof(uint32_t), m->lm.bigram_size, f);
    fwrite(m->lm.logp_bi, sizeof(int16_t), m->lm.bigram_size, f);
  }

  /* CRF features */
  if (m->crf.feat_count > 0) {
    if (!m->crf.feat_key || !m->crf.feat_w) {
      fclose(f);
      return -5;
    }
    fwrite(m->crf.feat_key, sizeof(uint32_t), m->crf.feat_count, f);
    fwrite(m->crf.feat_w, sizeof(int16_t), m->crf.feat_count, f);
  }

  /* v2: cc_ranges */
  if (m->cc.range_count > 0 && m->cc.ranges) {
    for (uint32_t i = 0; i < m->cc.range_count; i++) {
      wr_u32(f, m->cc.ranges[i].lo);
      wr_u32(f, m->cc.ranges[i].hi);
      uint8_t class_id = m->cc.ranges[i].class_id;
      fwrite(&class_id, 1, 1, f);
      uint8_t pad3[3] = {0, 0, 0};
      fwrite(pad3, 1, 3, f);
    }
  }

  fclose(f);
  return 0;
}

int mmjp_model_load_bin(const char *path, mmjp_loaded_model_t *out) {
  if (!path || !out) return -1;
  memset(out, 0, sizeof(*out));

  FILE *f = fopen(path, "rb");
  if (!f) return -10;

  /* magic */
  char magic[8];
  if (fread(magic, 1, 8, f) != 8) {
    fclose(f);
    return -11;
  }

  /* v1/v2 判定 */
  int is_v1 = 0;
  if (memcmp(magic, MMJP_MODEL_MAGIC, 8) == 0) {
    is_v1 = 0;  /* v2 */
  } else if (memcmp(magic, MMJP_MODEL_MAGIC_V1, 8) == 0) {
    is_v1 = 1;  /* v1 */
  } else {
    fclose(f);
    return -12;
  }

  uint32_t version = 0, da_index_bytes = 0, da_cap = 0, vocab = 0, max_word_len = 0;
  if (!rd_u32(f, &version) || !rd_u32(f, &da_index_bytes) || !rd_u32(f, &da_cap) || !rd_u32(f, &vocab) ||
      !rd_u32(f, &max_word_len)) {
    fclose(f);
    return -13;
  }
  if (is_v1) {
    if (version != MMJP_MODEL_VERSION_V1) {
      fclose(f);
      return -14;
    }
  } else {
    if (version != MMJP_MODEL_VERSION) {
      fclose(f);
      return -14;
    }
  }
  if (da_index_bytes != MMJP_DA_INDEX_BYTES) {
    fclose(f);
    return -15;
  }
  if (da_cap < 2 || vocab == 0 || max_word_len == 0) {
    fclose(f);
    return -16;
  }

  int16_t unk_base = 0, unk_per_cp = 0, lambda0 = 0;
  if (!rd_i16(f, &unk_base) || !rd_i16(f, &unk_per_cp) || !rd_i16(f, &lambda0)) {
    fclose(f);
    return -17;
  }

  int16_t trans00 = 0, trans01 = 0, trans10 = 0, trans11 = 0, bos_to1 = 0;
  if (!rd_i16(f, &trans00) || !rd_i16(f, &trans01) || !rd_i16(f, &trans10) || !rd_i16(f, &trans11) ||
      !rd_i16(f, &bos_to1)) {
    fclose(f);
    return -18;
  }

  uint32_t feat_count = 0, bigram_size = 0;
  if (!rd_u32(f, &feat_count) || !rd_u32(f, &bigram_size)) {
    fclose(f);
    return -19;
  }

  /* v2: flags, cc_mode, cc_fallback, cc_range_count */
  uint32_t flags = 0;
  uint8_t cc_mode = (uint8_t)NPYCRF_CC_MODE_UTF8LEN;
  uint8_t cc_fallback = (uint8_t)NPYCRF_CC_MODE_ASCII;
  uint32_t cc_range_count = 0;

  if (!is_v1) {
    if (!rd_u32(f, &flags)) {
      fclose(f);
      return -19;
    }
    uint8_t buf4[4];
    if (fread(buf4, 1, 4, f) != 4) {
      fclose(f);
      return -19;
    }
    cc_mode = buf4[0];
    cc_fallback = buf4[1];
    /* buf4[2], buf4[3] はパディング */
    if (!rd_u32(f, &cc_range_count)) {
      fclose(f);
      return -19;
    }
  }

  /* --- allocate owned block --- */
  /* base/check int32 -> da_index_t (assume int32 in CLI build) */
  size_t bytes = 0;
  bytes += (size_t)da_cap * sizeof(da_index_t);  /* base */
  bytes += (size_t)da_cap * sizeof(da_index_t);  /* check */
  bytes += (size_t)vocab * sizeof(int16_t);      /* unigram */
  bytes += (size_t)bigram_size * sizeof(uint32_t);
  bytes += (size_t)bigram_size * sizeof(int16_t);
  bytes += (size_t)feat_count * sizeof(uint32_t);
  bytes += (size_t)feat_count * sizeof(int16_t);
  bytes += (size_t)cc_range_count * sizeof(npycrf_cc_range_t);  /* v2: cc_ranges */

  uint8_t *mem = (uint8_t *)malloc(bytes);
  if (!mem) {
    fclose(f);
    return -20;
  }
  memset(mem, 0, bytes);

  uint8_t *p = mem;
  da_index_t *base = (da_index_t *)p;
  p += (size_t)da_cap * sizeof(da_index_t);
  da_index_t *check = (da_index_t *)p;
  p += (size_t)da_cap * sizeof(da_index_t);
  int16_t *unigram = (int16_t *)p;
  p += (size_t)vocab * sizeof(int16_t);
  uint32_t *bigram_key = NULL;
  int16_t *logp_bi = NULL;
  if (bigram_size > 0) {
    bigram_key = (uint32_t *)p;
    p += (size_t)bigram_size * sizeof(uint32_t);
    logp_bi = (int16_t *)p;
    p += (size_t)bigram_size * sizeof(int16_t);
  }
  uint32_t *feat_key = NULL;
  int16_t *feat_w = NULL;
  if (feat_count > 0) {
    feat_key = (uint32_t *)p;
    p += (size_t)feat_count * sizeof(uint32_t);
    feat_w = (int16_t *)p;
    p += (size_t)feat_count * sizeof(int16_t);
  }

  npycrf_cc_range_t *cc_ranges = NULL;
  if (cc_range_count > 0) {
    cc_ranges = (npycrf_cc_range_t *)p;
    p += (size_t)cc_range_count * sizeof(npycrf_cc_range_t);
  }

  /* --- read arrays --- */
  for (size_t i = 0; i < da_cap; i++) {
    uint32_t u = 0;
    if (!rd_u32(f, &u)) {
      free(mem);
      fclose(f);
      return -21;
    }
    base[i] = (da_index_t)(int32_t)u;
  }
  for (size_t i = 0; i < da_cap; i++) {
    uint32_t u = 0;
    if (!rd_u32(f, &u)) {
      free(mem);
      fclose(f);
      return -22;
    }
    check[i] = (da_index_t)(int32_t)u;
  }

  if (fread(unigram, sizeof(int16_t), vocab, f) != vocab) {
    free(mem);
    fclose(f);
    return -23;
  }

  if (bigram_size > 0) {
    if (fread(bigram_key, sizeof(uint32_t), bigram_size, f) != bigram_size) {
      free(mem);
      fclose(f);
      return -24;
    }
    if (fread(logp_bi, sizeof(int16_t), bigram_size, f) != bigram_size) {
      free(mem);
      fclose(f);
      return -25;
    }
  }

  if (feat_count > 0) {
    if (fread(feat_key, sizeof(uint32_t), feat_count, f) != feat_count) {
      free(mem);
      fclose(f);
      return -26;
    }
    if (fread(feat_w, sizeof(int16_t), feat_count, f) != feat_count) {
      free(mem);
      fclose(f);
      return -27;
    }
  }

  /* v2: cc_ranges */
  if (!is_v1 && cc_range_count > 0 && cc_ranges) {
    for (uint32_t i = 0; i < cc_range_count; i++) {
      uint32_t lo = 0, hi = 0;
      uint8_t buf4[4];
      if (!rd_u32(f, &lo) || !rd_u32(f, &hi)) {
        free(mem);
        fclose(f);
        return -28;
      }
      if (fread(buf4, 1, 4, f) != 4) {
        free(mem);
        fclose(f);
        return -28;
      }
      cc_ranges[i].lo = lo;
      cc_ranges[i].hi = hi;
      cc_ranges[i].class_id = buf4[0];
      cc_ranges[i]._pad[0] = 0;
      cc_ranges[i]._pad[1] = 0;
      cc_ranges[i]._pad[2] = 0;
    }
  }

  fclose(f);

  /* --- setup model pointers --- */
  npycrf_model_t m_out;
  memset(&m_out, 0, sizeof(m_out));

  m_out.max_word_len = (uint16_t)max_word_len;

  m_out.lm.trie.base = base;
  m_out.lm.trie.check = check;
  m_out.lm.trie.capacity = (size_t)da_cap;
  m_out.lm.logp_uni = unigram;
  m_out.lm.vocab_size = (uint32_t)vocab;
  m_out.lm.bigram_key = bigram_key;
  m_out.lm.logp_bi = logp_bi;
  m_out.lm.bigram_size = bigram_size;
  m_out.lm.unk_base = unk_base;
  m_out.lm.unk_per_cp = unk_per_cp;
  m_out.lambda0 = lambda0;

  m_out.crf.trans00 = trans00;
  m_out.crf.trans01 = trans01;
  m_out.crf.trans10 = trans10;
  m_out.crf.trans11 = trans11;
  m_out.crf.bos_to1 = bos_to1;
  m_out.crf.feat_key = feat_key;
  m_out.crf.feat_w = feat_w;
  m_out.crf.feat_count = feat_count;

  /* v2: flags and cc */
  m_out.flags = flags;
  m_out.cc.mode = (npycrf_cc_mode_t)cc_mode;
  m_out.cc.fallback = (npycrf_cc_mode_t)cc_fallback;
  m_out.cc.ranges = cc_ranges;
  m_out.cc.range_count = cc_range_count;

  out->m = m_out;
  out->cc_ranges_owned = cc_ranges;
  out->cc_ranges_count = cc_range_count;
  out->owned = mem;
  out->owned_bytes = bytes;
  return 0;
}

void mmjp_model_free(mmjp_loaded_model_t *m) {
  if (!m) return;
  free(m->owned);
  memset(m, 0, sizeof(*m));
}
