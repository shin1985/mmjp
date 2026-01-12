/*
 * mmjp_lossless.c
 *
 * 可逆トークナイズ（Lossless Tokenization）の実装
 */

#include "mmjp_lossless.h"
#include <string.h>

/* UTF-8エンコード（1コードポイント） */
static size_t utf8_encode_cp(uint32_t cp, uint8_t out[4]) {
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

/* UTF-8デコード（1コードポイント） */
static int utf8_decode_cp(const uint8_t *s, size_t len, size_t pos,
                          uint32_t *out_cp, size_t *out_adv) {
  if (pos >= len) return 0;
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
    *out_cp = ((uint32_t)(b0 & 0x0Fu) << 12) |
              ((uint32_t)(b1 & 0x3Fu) << 6) |
              (uint32_t)(b2 & 0x3Fu);
    *out_adv = 3;
    return 1;
  }
  if ((b0 & 0xF8u) == 0xF0u) {
    if (pos + 4 > len) return 0;
    uint8_t b1 = s[pos + 1], b2 = s[pos + 2], b3 = s[pos + 3];
    if ((b1 & 0xC0u) != 0x80u || (b2 & 0xC0u) != 0x80u || (b3 & 0xC0u) != 0x80u) return 0;
    *out_cp = ((uint32_t)(b0 & 0x07u) << 18) |
              ((uint32_t)(b1 & 0x3Fu) << 12) |
              ((uint32_t)(b2 & 0x3Fu) << 6) |
              (uint32_t)(b3 & 0x3Fu);
    *out_adv = 4;
    return 1;
  }
  return 0;
}

/* メタ文字かどうかをチェック */
static int is_meta_char(uint32_t cp) {
  return (cp >= MMJP_LOSSLESS_ESCAPE && cp <= MMJP_LOSSLESS_CR);
}

/*
 * losslessエンコード
 */
size_t mmjp_lossless_encode(const uint8_t *src, size_t src_len,
                            uint8_t *dst, size_t dst_cap,
                            int include_newlines) {
  if (!src && src_len > 0) return 0;

  size_t pos = 0;
  size_t out_len = 0;

  while (pos < src_len) {
    uint32_t cp = 0;
    size_t adv = 0;

    if (!utf8_decode_cp(src, src_len, pos, &cp, &adv)) {
      /* 無効なUTF-8はそのままコピー */
      if (dst && out_len < dst_cap) {
        dst[out_len] = src[pos];
      }
      out_len++;
      pos++;
      continue;
    }

    uint8_t enc[8];
    size_t enc_len = 0;

    if (cp == ' ') {
      /* スペース → ▁ */
      enc_len = utf8_encode_cp(MMJP_LOSSLESS_SPACE, enc);
    } else if (cp == '\t') {
      /* タブ → ▂ */
      enc_len = utf8_encode_cp(MMJP_LOSSLESS_TAB, enc);
    } else if (include_newlines && cp == '\n') {
      /* LF → ▃ */
      enc_len = utf8_encode_cp(MMJP_LOSSLESS_LF, enc);
    } else if (include_newlines && cp == '\r') {
      /* CR → ▄ */
      enc_len = utf8_encode_cp(MMJP_LOSSLESS_CR, enc);
    } else if (is_meta_char(cp)) {
      /* メタ文字はエスケープ: ▀ + 元文字 */
      enc_len = utf8_encode_cp(MMJP_LOSSLESS_ESCAPE, enc);
      enc_len += utf8_encode_cp(cp, enc + enc_len);
    } else {
      /* その他はそのままコピー */
      enc_len = adv;
      if (dst && out_len + enc_len <= dst_cap) {
        memcpy(dst + out_len, src + pos, enc_len);
      }
      out_len += enc_len;
      pos += adv;
      continue;
    }

    /* エンコード結果を出力 */
    if (dst && out_len + enc_len <= dst_cap) {
      memcpy(dst + out_len, enc, enc_len);
    }
    out_len += enc_len;
    pos += adv;
  }

  /* NUL終端 */
  if (dst && out_len < dst_cap) {
    dst[out_len] = '\0';
  }

  return out_len;
}

/*
 * losslessデコード
 */
size_t mmjp_lossless_decode(const uint8_t *src, size_t src_len,
                            uint8_t *dst, size_t dst_cap) {
  if (!src && src_len > 0) return 0;

  size_t pos = 0;
  size_t out_len = 0;

  while (pos < src_len) {
    uint32_t cp = 0;
    size_t adv = 0;

    if (!utf8_decode_cp(src, src_len, pos, &cp, &adv)) {
      /* 無効なUTF-8はそのままコピー */
      if (dst && out_len < dst_cap) {
        dst[out_len] = src[pos];
      }
      out_len++;
      pos++;
      continue;
    }

    uint8_t dec[4];
    size_t dec_len = 0;

    if (cp == MMJP_LOSSLESS_SPACE) {
      /* ▁ → スペース */
      dec[0] = ' ';
      dec_len = 1;
    } else if (cp == MMJP_LOSSLESS_TAB) {
      /* ▂ → タブ */
      dec[0] = '\t';
      dec_len = 1;
    } else if (cp == MMJP_LOSSLESS_LF) {
      /* ▃ → LF */
      dec[0] = '\n';
      dec_len = 1;
    } else if (cp == MMJP_LOSSLESS_CR) {
      /* ▄ → CR */
      dec[0] = '\r';
      dec_len = 1;
    } else if (cp == MMJP_LOSSLESS_ESCAPE) {
      /* ▀ + X → X (エスケープ解除) */
      pos += adv;
      if (pos < src_len) {
        uint32_t next_cp = 0;
        size_t next_adv = 0;
        if (utf8_decode_cp(src, src_len, pos, &next_cp, &next_adv)) {
          dec_len = utf8_encode_cp(next_cp, dec);
          adv = next_adv;
        } else {
          /* エスケープ後が無効ならエスケープ文字をそのまま出力 */
          dec_len = utf8_encode_cp(MMJP_LOSSLESS_ESCAPE, dec);
          adv = 0;
        }
      } else {
        /* 末尾のエスケープ文字はそのまま出力 */
        dec_len = utf8_encode_cp(MMJP_LOSSLESS_ESCAPE, dec);
        adv = 0;
      }
    } else {
      /* その他はそのままコピー */
      if (dst && out_len + adv <= dst_cap) {
        memcpy(dst + out_len, src + pos, adv);
      }
      out_len += adv;
      pos += adv;
      continue;
    }

    /* デコード結果を出力 */
    if (dst && out_len + dec_len <= dst_cap) {
      memcpy(dst + out_len, dec, dec_len);
    }
    out_len += dec_len;
    pos += adv;
  }

  /* NUL終端 */
  if (dst && out_len < dst_cap) {
    dst[out_len] = '\0';
  }

  return out_len;
}

/*
 * detokenize用ヘルパー
 */
size_t mmjp_lossless_detokenize(const uint8_t **tokens, const size_t *token_lens,
                                size_t n_tokens,
                                uint8_t *dst, size_t dst_cap) {
  if (!tokens || !token_lens) return 0;

  /* まずトークンを結合（区切りスペースは含めない） */
  size_t total_len = 0;
  for (size_t i = 0; i < n_tokens; i++) {
    total_len += token_lens[i];
  }

  /* 一時バッファを使わず、直接デコード */
  /* まず結合後のサイズを計算するためにNULLで呼ぶ */

  size_t out_len = 0;

  for (size_t i = 0; i < n_tokens; i++) {
    if (!tokens[i] || token_lens[i] == 0) continue;

    /* トークンをデコードして出力 */
    size_t decoded = mmjp_lossless_decode(tokens[i], token_lens[i],
                                          dst ? dst + out_len : NULL,
                                          dst ? dst_cap - out_len : 0);
    out_len += decoded;
  }

  /* NUL終端 */
  if (dst && out_len < dst_cap) {
    dst[out_len] = '\0';
  }

  return out_len;
}
