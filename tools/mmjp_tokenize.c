/*
 * mmjp_tokenize.c
 *
 * 学習済み MMJP (NPYCRF Lite) モデルを使って分かち書き。
 *  - 標準入力から 1行=1文 として読み、トークンを空白区切りで出力
 *  - または引数に文字列を渡して1回だけ実行
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mmjp_model.h"
#include "../mmjp_lossless.h"


/* =====================
 * UTF-8 normalize (CLI side)
 *  - normalize to canonical UTF-8 (avoid overlong sequences)
 *  - invalid sequences -> fallback char
 *
 * NOTE: This is mainly to make it possible to tokenize imperfect corpora dumps.
 * For normal UTF-8 input, output is identical.
 * ===================== */

static int utf8_decode1_cli(const uint8_t *s, size_t len, size_t pos, uint32_t *out_cp, size_t *out_adv) {
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

static size_t utf8_encode1_cli(uint32_t cp, uint8_t out[4]) {
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

static int normalize_utf8_canonical(const uint8_t *in, size_t in_len, uint32_t fallback_cp,
                                   uint8_t **out, size_t *out_cap, size_t *out_len) {
  if (!in || !out || !out_cap || !out_len) return 0;
  uint8_t fb[4];
  size_t fb_len = utf8_encode1_cli(fallback_cp, fb);

  if (*out_cap < in_len + 1) {
    size_t nc = (*out_cap == 0) ? 256 : (*out_cap);
    while (nc < in_len + 1) nc *= 2;
    uint8_t *nb = (uint8_t *)realloc(*out, nc);
    if (!nb) return 0;
    *out = nb;
    *out_cap = nc;
  }

  size_t pos = 0;
  size_t op = 0;
  while (pos < in_len) {
    uint32_t cp = 0;
    size_t adv = 0;
    int ok = utf8_decode1_cli(in, in_len, pos, &cp, &adv);
    if (!ok || adv == 0) {
      if (op + fb_len + 1 > *out_cap) {
        size_t nc = (*out_cap) * 2;
        while (nc < op + fb_len + 1) nc *= 2;
        uint8_t *nb = (uint8_t *)realloc(*out, nc);
        if (!nb) return 0;
        *out = nb;
        *out_cap = nc;
      }
      memcpy(*out + op, fb, fb_len);
      op += fb_len;
      pos += 1;
      continue;
    }

    /* invalid Unicode scalar -> fallback */
    if (cp > 0x10FFFFu || (cp >= 0xD800u && cp <= 0xDFFFu)) {
      if (op + fb_len + 1 > *out_cap) {
        size_t nc = (*out_cap) * 2;
        while (nc < op + fb_len + 1) nc *= 2;
        uint8_t *nb = (uint8_t *)realloc(*out, nc);
        if (!nb) return 0;
        *out = nb;
        *out_cap = nc;
      }
      memcpy(*out + op, fb, fb_len);
      op += fb_len;
      pos += adv;
      continue;
    }

    /* canonical encode (this also removes overlong sequences) */
    uint8_t enc[4];
    size_t enc_len = utf8_encode1_cli(cp, enc);
    if (op + enc_len + 1 > *out_cap) {
      size_t nc = (*out_cap) * 2;
      while (nc < op + enc_len + 1) nc *= 2;
      uint8_t *nb = (uint8_t *)realloc(*out, nc);
      if (!nb) return 0;
      *out = nb;
      *out_cap = nc;
    }
    memcpy(*out + op, enc, enc_len);
    op += enc_len;
    pos += adv;
  }

  (*out)[op] = 0;
  *out_len = op;
  return 1;
}

static void usage(const char *prog) {
  fprintf(stderr,
          "Usage: %s --model model.bin [options] [text...]\n"
          "  If text is not given, read stdin lines.\n"
          "\n",
          prog);
  fprintf(stderr,
          "Options:\n"
          "  --max_n_cp N          workspace max codepoints (default: 1024)\n"
          "  --max_line_bytes N    skip lines longer than this (default: 16384)\n"
          "  --no_normalize        do not normalize UTF-8 (CLI side)\n"
          "  --fallback_char C     fallback ASCII char for invalid UTF-8 (default: ?)\n"
          "\n"
          "Lossless tokenization:\n"
          "  --lossless_ws N       -1=auto (from model), 0=off, 1=on (default: -1)\n"
          "  --read_all 1          read all stdin as one text (include newlines)\n"
          "  --detok               detokenize mode (token stream -> original text)\n"
          "\n"
          "Stochastic tokenization (Subword Regularization):\n"
          "  --sample              FFBS sampling (one sample)\n"
          "  --temperature X       sampling temperature (default: 1.0)\n"
          "  --seed N              RNG seed (default: 1)\n"
          "\n"
          "N-best Viterbi:\n"
          "  --nbest N             output N-best segmentations (one per line)\n"
          "  --sample_nbest N      sample 1 segmentation from top-N (uniform)\n"
          "\n"
          "Notes:\n"
          "  - --sample / --sample_nbest are intended for dataset augmentation.\n"
          "  - --nbest is mainly for debugging/analysis.\n"
          "  - --lossless_ws 1 encodes spaces as meta-chars for lossless round-trip.\n"
          "  - --detok restores original text from lossless token stream.\n");
}

typedef enum {
  MODE_BEST = 0,
  MODE_SAMPLE_FFBS = 1,
  MODE_NBEST_LIST = 2,
  MODE_SAMPLE_NBEST = 3,
} decode_mode_t;

static inline uint32_t xs32(uint32_t *s) {
  uint32_t x = (s && *s) ? *s : 0x12345678u;
  x ^= x << 13;
  x ^= x >> 17;
  x ^= x << 5;
  if (s) *s = x;
  return x;
}

static int read_line_dynamic(FILE *f, char **buf, size_t *cap, size_t *len, size_t max_bytes) {
  if (!f || !buf || !cap || !len) return 0;
  *len = 0;
  int c;
  while ((c = fgetc(f)) != EOF) {
    if (c == '\n') break;
    if (max_bytes > 0 && *len >= max_bytes) {
      /* discard rest */
      while ((c = fgetc(f)) != EOF && c != '\n') {}
      *len = 0;
      return 1;
    }
    if (*len + 1 > *cap) {
      size_t nc = (*cap == 0) ? 256 : (*cap * 2);
      char *nb = (char *)realloc(*buf, nc);
      if (!nb) return 0;
      *buf = nb;
      *cap = nc;
    }
    (*buf)[(*len)++] = (char)c;
  }
  if (c == EOF && *len == 0) return 0;
  /* trim CR */
  while (*len > 0 && ((*buf)[*len - 1] == '\r')) (*len)--;
  if (*len + 1 > *cap) {
    char *nb = (char *)realloc(*buf, *len + 1);
    if (!nb) return 0;
    *buf = nb;
    *cap = *len + 1;
  }
  (*buf)[*len] = '\0';
  return 1;
}

static int tokenize_one(mmjp_loaded_model_t *mb, const uint8_t *utf8, size_t len,
                        uint8_t **workbuf, size_t *workcap,
                        npycrf_work_t *wk,
                        uint16_t **b_cp, size_t *bcp_cap,
                        uint16_t **b_bytes, size_t *bb_cap,
                        /* stochastic/nbest buffers */
                        uint8_t **samplebuf, size_t *samplecap,
                        uint8_t **nbestbuf, size_t *nbestcap,
                        uint16_t **bcp_flat, size_t *bcp_flat_cap,
                        size_t **bcount_arr, size_t *bcount_cap,
                        npycrf_score_t **score_arr, size_t *score_cap,
                        decode_mode_t mode,
                        uint16_t nbest,
                        double temperature,
                        uint32_t *seed_io,
                        size_t *max_n_cp_inout) {
  if (!mb || !utf8 || !wk || !workbuf || !workcap || !b_cp || !b_bytes || !max_n_cp_inout) return 0;

  /* work buffer (npycrf_work_t) */
  size_t max_n_cp = *max_n_cp_inout;
  for (;;) {
    size_t need = npycrf_workbuf_size(max_n_cp, mb->m.max_word_len);
    if (*workcap < need) {
      uint8_t *nb = (uint8_t *)realloc(*workbuf, need);
      if (!nb) return 0;
      *workbuf = nb;
      *workcap = need;
    }
    if (npycrf_work_init(wk, *workbuf, *workcap, (uint16_t)max_n_cp, mb->m.max_word_len) != 0) {
      return 0;
    }
    /* boundary buffers */
    size_t out_cap = max_n_cp + 1u;
    if (*bcp_cap < out_cap) {
      uint16_t *nb = (uint16_t *)realloc(*b_cp, out_cap * sizeof(uint16_t));
      if (!nb) return 0;
      *b_cp = nb;
      *bcp_cap = out_cap;
    }
    if (*bb_cap < out_cap) {
      uint16_t *nb = (uint16_t *)realloc(*b_bytes, out_cap * sizeof(uint16_t));
      if (!nb) return 0;
      *b_bytes = nb;
      *bb_cap = out_cap;
    }

    size_t b_count = 0;
    npycrf_score_t score = 0;
    int rc = 0;

    if (mode == MODE_SAMPLE_FFBS) {
      /* ensure sample buffer */
      size_t need_s = npycrf_samplebuf_size((uint16_t)max_n_cp, mb->m.max_word_len);
      if (*samplecap < need_s) {
        uint8_t *nb = (uint8_t *)realloc(*samplebuf, need_s);
        if (!nb) return 0;
        *samplebuf = nb;
        *samplecap = need_s;
      }
      uint32_t seed = seed_io ? *seed_io : 1u;
      rc = npycrf_decode_sample(&mb->m, utf8, len, wk,
                                *samplebuf, *samplecap,
                                temperature,
                                seed,
                                *b_cp, *bcp_cap, &b_count,
                                &score);
      if (seed_io) *seed_io = xs32(seed_io);
    } else if (mode == MODE_NBEST_LIST || mode == MODE_SAMPLE_NBEST) {
      if (nbest == 0) nbest = 1;

      /* ensure nbest work buffer */
      size_t need_n = npycrf_nbestbuf_size((uint16_t)max_n_cp, mb->m.max_word_len, nbest);
      if (*nbestcap < need_n) {
        uint8_t *nb = (uint8_t *)realloc(*nbestbuf, need_n);
        if (!nb) return 0;
        *nbestbuf = nb;
        *nbestcap = need_n;
      }

      /* ensure flat boundary storage */
      size_t per = max_n_cp + 1u;
      size_t flat_need = (size_t)nbest * per;
      if (*bcp_flat_cap < flat_need) {
        uint16_t *nb = (uint16_t *)realloc(*bcp_flat, flat_need * sizeof(uint16_t));
        if (!nb) return 0;
        *bcp_flat = nb;
        *bcp_flat_cap = flat_need;
      }
      if (*bcount_cap < (size_t)nbest) {
        size_t *nb = (size_t *)realloc(*bcount_arr, (size_t)nbest * sizeof(size_t));
        if (!nb) return 0;
        *bcount_arr = nb;
        *bcount_cap = (size_t)nbest;
      }
      if (*score_cap < (size_t)nbest) {
        npycrf_score_t *nb = (npycrf_score_t *)realloc(*score_arr, (size_t)nbest * sizeof(npycrf_score_t));
        if (!nb) return 0;
        *score_arr = nb;
        *score_cap = (size_t)nbest;
      }

      int n_out = npycrf_decode_nbest(&mb->m, utf8, len, wk,
                                     *nbestbuf, *nbestcap,
                                     nbest,
                                     *bcp_flat, per,
                                     *bcount_arr,
                                     *score_arr);
      if (n_out < 0) {
        fprintf(stderr, "npycrf_decode_nbest failed rc=%d\n", n_out);
        return 0;
      }
      if (n_out == 0) {
        /* fallback to best */
        rc = npycrf_decode(&mb->m, utf8, len, wk, *b_cp, *bcp_cap, &b_count, &score);
      } else {
        if (mode == MODE_SAMPLE_NBEST) {
          /* choose one candidate uniformly from available */
          uint32_t seed = seed_io ? *seed_io : 1u;
          uint32_t r = xs32(&seed);
          if (seed_io) *seed_io = seed;
          int pick = (int)(r % (uint32_t)n_out);
          /* copy picked boundaries into b_cp */
          size_t pcnt = (*bcount_arr)[(size_t)pick];
          if (pcnt > *bcp_cap) return 0;
          memcpy(*b_cp, *bcp_flat + (size_t)pick * per, pcnt * sizeof(uint16_t));
          b_count = pcnt;
          score = (*score_arr)[(size_t)pick];
          rc = 0;
        } else {
          /* list mode: print each candidate, then return */
          for (int ci = 0; ci < n_out; ci++) {
            size_t pcnt = (*bcount_arr)[(size_t)ci];
            if (pcnt < 2) continue;
            npycrf_boundaries_cp_to_bytes(wk->cp_off, *bcp_flat + (size_t)ci * per, pcnt, *b_bytes);
            for (size_t i = 0; i + 1 < pcnt; i++) {
              uint16_t s = (*b_bytes)[i];
              uint16_t e = (*b_bytes)[i + 1];
              if (e > len) e = (uint16_t)len;
              if (s > e) s = e;
              fwrite(utf8 + s, 1, (size_t)(e - s), stdout);
              if (i + 2 < pcnt) fputc(' ', stdout);
            }
            fputc('\n', stdout);
          }
          *max_n_cp_inout = max_n_cp;
          return 1;
        }
      }
    } else {
      rc = npycrf_decode(&mb->m, utf8, len, wk, *b_cp, *bcp_cap, &b_count, &score);
    }
    if (rc == -3) {
      /* cp_off overflow -> grow max_n_cp */
      max_n_cp = max_n_cp * 2u;
      if (max_n_cp > 65530u) return 0;
      continue;
    }
    if (rc != 0) {
      fprintf(stderr, "npycrf_decode failed rc=%d\n", rc);
      return 0;
    }

    /* convert cp boundaries -> byte boundaries */
    npycrf_boundaries_cp_to_bytes(wk->cp_off, *b_cp, b_count, *b_bytes);

    /* print tokens */
    for (size_t i = 0; i + 1 < b_count; i++) {
      uint16_t s = (*b_bytes)[i];
      uint16_t e = (*b_bytes)[i + 1];
      if (e > len) e = (uint16_t)len;
      if (s > e) s = e;
      fwrite(utf8 + s, 1, (size_t)(e - s), stdout);
      if (i + 2 < b_count) fputc(' ', stdout);
    }
    fputc('\n', stdout);

    *max_n_cp_inout = max_n_cp;
    return 1;
  }
}

int main(int argc, char **argv) {
  const char *model_path = NULL;
  size_t max_n_cp = 1024u;
  size_t max_line_bytes = 16384u;
  int normalize = 1;
  uint32_t fallback_cp = '?';

  decode_mode_t mode = MODE_BEST;
  uint16_t nbest = 8;
  double temperature = 1.0;
  uint32_t seed = 1u;

  unsigned nsamples = 1u;

  /* lossless options */
  int lossless_ws = -1;  /* -1=auto, 0=off, 1=on */
  int read_all = 0;
  int detok_mode = 0;

  int argi = 1;
  for (; argi < argc; argi++) {
    if (strcmp(argv[argi], "--model") == 0 && argi + 1 < argc) {
      model_path = argv[++argi];
    } else if (strcmp(argv[argi], "--max_n_cp") == 0 && argi + 1 < argc) {
      max_n_cp = (size_t)strtoull(argv[++argi], NULL, 10);
    } else if (strcmp(argv[argi], "--max_line_bytes") == 0 && argi + 1 < argc) {
      max_line_bytes = (size_t)strtoull(argv[++argi], NULL, 10);
    } else if (strcmp(argv[argi], "--no_normalize") == 0) {
      normalize = 0;
    } else if (strcmp(argv[argi], "--fallback_char") == 0 && argi + 1 < argc) {
      const char *fc = argv[++argi];
      /* take first byte as ASCII fallback (recommended: '?') */
      fallback_cp = (uint8_t)fc[0];
    } else if (strcmp(argv[argi], "--lossless_ws") == 0 && argi + 1 < argc) {
      lossless_ws = (int)strtol(argv[++argi], NULL, 10);
    } else if (strcmp(argv[argi], "--read_all") == 0 && argi + 1 < argc) {
      read_all = (int)strtol(argv[++argi], NULL, 10);
    } else if (strcmp(argv[argi], "--detok") == 0) {
      detok_mode = 1;
    } else if (strcmp(argv[argi], "--sample") == 0) {
      mode = MODE_SAMPLE_FFBS;
    } else if (strcmp(argv[argi], "--temperature") == 0 && argi + 1 < argc) {
      temperature = strtod(argv[++argi], NULL);
    } else if (strcmp(argv[argi], "--seed") == 0 && argi + 1 < argc) {
      seed = (uint32_t)strtoul(argv[++argi], NULL, 10);
    } else if (strcmp(argv[argi], "--nsamples") == 0 && argi + 1 < argc) {
      nsamples = (unsigned)strtoul(argv[++argi], NULL, 10);
      if (nsamples == 0u) nsamples = 1u;
    } else if (strcmp(argv[argi], "--nbest") == 0 && argi + 1 < argc) {
      mode = MODE_NBEST_LIST;
      nbest = (uint16_t)strtoul(argv[++argi], NULL, 10);
      if (nbest == 0) nbest = 1;
    } else if (strcmp(argv[argi], "--sample_nbest") == 0 && argi + 1 < argc) {
      mode = MODE_SAMPLE_NBEST;
      nbest = (uint16_t)strtoul(argv[++argi], NULL, 10);
      if (nbest == 0) nbest = 1;
    } else if (strcmp(argv[argi], "-h") == 0 || strcmp(argv[argi], "--help") == 0) {
      usage(argv[0]);
      return 0;
    } else {
      break;
    }
  }

  if (!model_path) {
    usage(argv[0]);
    return 1;
  }

  mmjp_loaded_model_t mb;
  int rc = mmjp_model_load_bin(model_path, &mb);
  if (rc != 0) {
    fprintf(stderr, "failed to load model rc=%d\n", rc);
    return 1;
  }

  /* auto-detect lossless_ws from model flags */
  if (lossless_ws == -1) {
    lossless_ws = (mb.m.flags & NPYCRF_FLAG_LOSSLESS_WS) ? 1 : 0;
  }

  /* detokenize mode: read token stream, decode lossless, output original text */
  if (detok_mode) {
    char *line = NULL;
    size_t cap = 0;
    size_t len = 0;

    /* line-by-line detokenization for proper roundtrip */
    uint8_t *concat_buf = NULL;
    size_t concat_cap = 0;

    while (read_line_dynamic(stdin, &line, &cap, &len, max_line_bytes)) {
      /* split by spaces and concatenate tokens for this line */
      size_t concat_len = 0;
      size_t pos = 0;
      while (pos < len) {
        /* skip spaces (token separators) */
        while (pos < len && (line[pos] == ' ' || line[pos] == '\t')) pos++;
        if (pos >= len) break;

        /* find token end */
        size_t tok_start = pos;
        while (pos < len && line[pos] != ' ' && line[pos] != '\t') pos++;
        size_t tok_len = pos - tok_start;

        /* append to concat buffer */
        if (concat_len + tok_len + 1 > concat_cap) {
          size_t new_cap = concat_cap ? concat_cap * 2 : 1024;
          while (new_cap < concat_len + tok_len + 1) new_cap *= 2;
          uint8_t *new_buf = (uint8_t *)realloc(concat_buf, new_cap);
          if (!new_buf) {
            free(concat_buf);
            free(line);
            mmjp_model_free(&mb);
            return 1;
          }
          concat_buf = new_buf;
          concat_cap = new_cap;
        }
        memcpy(concat_buf + concat_len, line + tok_start, tok_len);
        concat_len += tok_len;
      }

      if (concat_len > 0) {
        /* decode lossless for this line */
        size_t dec_len = mmjp_lossless_decode(concat_buf, concat_len, NULL, 0);
        uint8_t *dec_buf = (uint8_t *)malloc(dec_len + 1);
        if (dec_buf) {
          mmjp_lossless_decode(concat_buf, concat_len, dec_buf, dec_len + 1);
          fwrite(dec_buf, 1, dec_len, stdout);
          /* only add newline if decoded content doesn't end with newline */
          if (dec_len == 0 || dec_buf[dec_len - 1] != '\n') {
            fputc('\n', stdout);
          }
          free(dec_buf);
        }
      } else {
        /* empty line - output newline to preserve line structure */
        fputc('\n', stdout);
      }
    }
    free(line);
    free(concat_buf);

    mmjp_model_free(&mb);
    return 0;
  }

  uint8_t *workbuf = NULL;
  size_t workcap = 0;

  uint8_t *norm = NULL;
  size_t norm_cap = 0;
  npycrf_work_t wk;
  memset(&wk, 0, sizeof(wk));

  uint16_t *bcp = NULL;
  size_t bcp_cap = 0;
  uint16_t *bbytes = NULL;
  size_t bb_cap = 0;

  /* stochastic/nbest extra buffers */
  uint8_t *samplebuf = NULL;
  size_t samplecap = 0;
  uint8_t *nbestbuf = NULL;
  size_t nbestcap = 0;
  uint16_t *bcp_flat = NULL;
  size_t bcp_flat_cap = 0;
  size_t *bcount_arr = NULL;
  size_t bcount_cap = 0;
  npycrf_score_t *score_arr = NULL;
  size_t score_cap = 0;

  /* lossless encode buffer */
  uint8_t *lossless_buf = NULL;
  size_t lossless_cap = 0;

  /* read_all mode: read all stdin as one text */
  if (read_all && argi >= argc) {
    /* read entire stdin */
    uint8_t *all_buf = NULL;
    size_t all_cap = 0;
    size_t all_len = 0;
    int c;
    while ((c = fgetc(stdin)) != EOF) {
      if (all_len + 1 > all_cap) {
        size_t new_cap = all_cap ? all_cap * 2 : 4096;
        uint8_t *new_buf = (uint8_t *)realloc(all_buf, new_cap);
        if (!new_buf) {
          free(all_buf);
          mmjp_model_free(&mb);
          return 1;
        }
        all_buf = new_buf;
        all_cap = new_cap;
      }
      all_buf[all_len++] = (uint8_t)c;
    }

    if (all_buf && all_len > 0) {
      const uint8_t *inp = all_buf;
      size_t inlen = all_len;

      /* lossless encode if enabled (include newlines in read_all mode) */
      if (lossless_ws > 0) {
        size_t enc_len = mmjp_lossless_encode(inp, inlen, NULL, 0, 1);
        if (enc_len + 1 > lossless_cap) {
          lossless_cap = enc_len + 1;
          lossless_buf = (uint8_t *)realloc(lossless_buf, lossless_cap);
        }
        if (lossless_buf) {
          mmjp_lossless_encode(inp, inlen, lossless_buf, lossless_cap, 1);
          inp = lossless_buf;
          inlen = enc_len;
        }
      }

      if (normalize) {
        size_t nlen = 0;
        if (!normalize_utf8_canonical(inp, inlen, fallback_cp, &norm, &norm_cap, &nlen)) {
          fprintf(stderr, "normalize failed\n");
          free(all_buf);
          mmjp_model_free(&mb);
          return 1;
        }
        inp = norm;
        inlen = nlen;
      }

      tokenize_one(&mb, inp, inlen,
                   &workbuf, &workcap, &wk,
                   &bcp, &bcp_cap, &bbytes, &bb_cap,
                   &samplebuf, &samplecap,
                   &nbestbuf, &nbestcap,
                   &bcp_flat, &bcp_flat_cap,
                   &bcount_arr, &bcount_cap,
                   &score_arr, &score_cap,
                   mode, nbest, temperature, &seed,
                   &max_n_cp);
    }
    free(all_buf);
    goto cleanup;
  }

  /* if text args exist, join them with spaces as one line */
  if (argi < argc) {
    size_t total = 0;
    for (int i = argi; i < argc; i++) total += strlen(argv[i]) + 1;
    char *line = (char *)malloc(total + 1);
    if (!line) {
      mmjp_model_free(&mb);
      return 1;
    }
    line[0] = '\0';
    for (int i = argi; i < argc; i++) {
      strcat(line, argv[i]);
      if (i + 1 < argc) strcat(line, " ");
    }
    const uint8_t *inp = (const uint8_t *)line;
    size_t inlen = strlen(line);

    /* lossless encode if enabled */
    if (lossless_ws > 0) {
      size_t enc_len = mmjp_lossless_encode(inp, inlen, NULL, 0, 0);
      if (enc_len + 1 > lossless_cap) {
        lossless_cap = enc_len + 1;
        lossless_buf = (uint8_t *)realloc(lossless_buf, lossless_cap);
      }
      if (lossless_buf) {
        mmjp_lossless_encode(inp, inlen, lossless_buf, lossless_cap, 0);
        inp = lossless_buf;
        inlen = enc_len;
      }
    }

    if (normalize) {
      size_t nlen = 0;
      if (!normalize_utf8_canonical(inp, inlen, fallback_cp, &norm, &norm_cap, &nlen)) {
        fprintf(stderr, "normalize failed\n");
        free(line);
        mmjp_model_free(&mb);
        return 1;
      }
      inp = norm;
      inlen = nlen;
    }
    unsigned reps = 1u;
    if (mode == MODE_SAMPLE_FFBS || mode == MODE_SAMPLE_NBEST) reps = nsamples;
    for (unsigned r = 0; r < reps; r++) {
      tokenize_one(&mb, inp, inlen,
                   &workbuf, &workcap, &wk,
                   &bcp, &bcp_cap, &bbytes, &bb_cap,
                   &samplebuf, &samplecap,
                   &nbestbuf, &nbestcap,
                   &bcp_flat, &bcp_flat_cap,
                   &bcount_arr, &bcount_cap,
                   &score_arr, &score_cap,
                   mode, nbest, temperature, &seed,
                   &max_n_cp);
    }
    free(line);
  } else {
    char *line = NULL;
    size_t cap = 0;
    size_t len = 0;
    while (read_line_dynamic(stdin, &line, &cap, &len, max_line_bytes)) {
      if (len == 0) continue;
      const uint8_t *inp = (const uint8_t *)line;
      size_t inlen = len;

      /* lossless encode if enabled */
      if (lossless_ws > 0) {
        size_t enc_len = mmjp_lossless_encode(inp, inlen, NULL, 0, 0);
        if (enc_len + 1 > lossless_cap) {
          lossless_cap = enc_len + 1;
          lossless_buf = (uint8_t *)realloc(lossless_buf, lossless_cap);
        }
        if (lossless_buf) {
          mmjp_lossless_encode(inp, inlen, lossless_buf, lossless_cap, 0);
          inp = lossless_buf;
          inlen = enc_len;
        }
      }

      if (normalize) {
        size_t nlen = 0;
        if (!normalize_utf8_canonical(inp, inlen, fallback_cp, &norm, &norm_cap, &nlen)) {
          fprintf(stderr, "normalize failed\n");
          break;
        }
        inp = norm;
        inlen = nlen;
      }
      unsigned reps = 1u;
      if (mode == MODE_SAMPLE_FFBS || mode == MODE_SAMPLE_NBEST) reps = nsamples;
      for (unsigned r = 0; r < reps; r++) {
        tokenize_one(&mb, inp, inlen,
                     &workbuf, &workcap, &wk,
                     &bcp, &bcp_cap, &bbytes, &bb_cap,
                     &samplebuf, &samplecap,
                     &nbestbuf, &nbestcap,
                     &bcp_flat, &bcp_flat_cap,
                     &bcount_arr, &bcount_cap,
                     &score_arr, &score_cap,
                     mode, nbest, temperature, &seed,
                     &max_n_cp);
      }
    }
    free(line);
  }

cleanup:
  free(workbuf);
  free(norm);
  free(bcp);
  free(bbytes);
  free(samplebuf);
  free(nbestbuf);
  free(bcp_flat);
  free(bcount_arr);
  free(score_arr);
  free(lossless_buf);
  mmjp_model_free(&mb);
  return 0;
}
