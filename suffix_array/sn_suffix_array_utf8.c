/*
 * sn_suffix_array_utf8.c
 *
 * SuffixArray 互換ラッパーの実装
 */

#include "sn_suffix_array_utf8.h"

#include <string.h>
#include <stdio.h>

/* メモリ上のテキストから接尾辞配列を構築 */
int sn_sa_build(sn_suffix_array_t *obj,
                const char *name,
                const uint8_t *text, size_t text_len,
                sa_idx_t *sa_buf, size_t sa_buf_cap,
                unsigned build_flags) {
  if (!obj || !text || !sa_buf || sa_buf_cap == 0) return 0;
  memset(obj, 0, sizeof(*obj));
  if (name) {
    /* NUL 終端を保証 */
    strncpy(obj->name, name, sizeof(obj->name) - 1);
  }
  obj->text = text;
  obj->text_len = text_len;
  obj->sa = sa_buf;
  obj->sa_cap = sa_buf_cap;
  obj->build_flags = build_flags;

  const size_t n = sa_utf8_build(sa_buf, sa_buf_cap, text, text_len, build_flags);
  if (n == 0) {
    obj->sa_len = 0;
    return 0;
  }
  obj->sa_len = n;
  return 1;
}

/* キーワードの出現数を取得 */
size_t sn_sa_get_count(const sn_suffix_array_t *obj, const char *keyword) {
  if (!obj || !keyword) return 0;
  const size_t klen = strlen(keyword);
  sa_utf8_view_t view = sa_utf8_view(obj->text, obj->text_len, obj->sa, obj->sa_len);
  return sa_utf8_count_prefix(&view, (const uint8_t *)keyword, klen);
}

/* バイグラム出現数を取得 */
sa_bigram_count_t sn_sa_get_bigram_count(const sn_suffix_array_t *obj,
                                         const char *forward_word,
                                         const char *back_word) {
  sa_bigram_count_t z = {0, 0};
  if (!obj || !forward_word) return z;
  const size_t flen = strlen(forward_word);
  const size_t blen = back_word ? strlen(back_word) : 0;
  sa_utf8_view_t view = sa_utf8_view(obj->text, obj->text_len, obj->sa, obj->sa_len);
  return sa_utf8_count_bigram(&view,
                              (const uint8_t *)forward_word, flen,
                              (const uint8_t *)back_word, blen);
}

/* N-gram デバッグ表示 */
void sn_sa_show_ngram(const sn_suffix_array_t *obj, size_t n_codepoints) {
  if (!obj || !obj->text || !obj->sa || obj->sa_len == 0) return;
  sa_utf8_view_t view = sa_utf8_view(obj->text, obj->text_len, obj->sa, obj->sa_len);

  char last[256];
  last[0] = '\0';
  char cur[256];
  size_t count = 0;

  for (size_t i = 0; i < view.sa_len; ++i) {
    const size_t pos = (size_t)view.sa[i];
    sa_utf8_copy_prefix_n(view.text, view.text_len, pos, n_codepoints, cur, sizeof(cur), obj->build_flags);

    if (strcmp(last, cur) != 0) {
      if (i != 0) {
        printf("%s,%lu\n", last, (unsigned long)count);
      }
      strncpy(last, cur, sizeof(last) - 1);
      last[sizeof(last) - 1] = '\0';
      count = 0;
    }
    ++count;
  }
  if (view.sa_len > 0) {
    printf("%s,%lu\n", last, (unsigned long)count);
  }
}

#ifdef SN_SA_ENABLE_FILEIO
/*
 * シンプルなバイナリ形式:
 * [u32 text_len][bytes text][u32 sa_len][sa_idx_t[]]
 */

/* 32 ビット値の書き込み */
static int sn_write_u32(FILE *fp, uint32_t v) {
  return (fwrite(&v, sizeof(v), 1, fp) == 1);
}

/* 32 ビット値の読み込み */
static int sn_read_u32(FILE *fp, uint32_t *out) {
  return (fread(out, sizeof(*out), 1, fp) == 1);
}

/* ファイルに保存 */
int sn_sa_save(const sn_suffix_array_t *obj, const char *path) {
  if (!obj || !path || !obj->text || !obj->sa) return 0;
  FILE *fp = fopen(path, "wb");
  if (!fp) return 0;

  const uint32_t tlen = (uint32_t)obj->text_len;
  const uint32_t slen = (uint32_t)obj->sa_len;
  int ok = 1;
  ok &= sn_write_u32(fp, tlen);
  ok &= (fwrite(obj->text, 1, obj->text_len, fp) == obj->text_len);
  ok &= sn_write_u32(fp, slen);
  ok &= (fwrite(obj->sa, sizeof(sa_idx_t), obj->sa_len, fp) == obj->sa_len);
  fclose(fp);
  return ok;
}

/* ファイルから読み込み */
int sn_sa_load(sn_suffix_array_t *obj,
               const char *path,
               uint8_t *text_buf, size_t text_buf_cap,
               sa_idx_t *sa_buf, size_t sa_buf_cap,
               unsigned build_flags) {
  if (!obj || !path || !text_buf || !sa_buf) return 0;
  FILE *fp = fopen(path, "rb");
  if (!fp) return 0;

  uint32_t tlen = 0, slen = 0;
  if (!sn_read_u32(fp, &tlen)) { fclose(fp); return 0; }
  if ((size_t)tlen > text_buf_cap) { fclose(fp); return 0; }
  if (fread(text_buf, 1, tlen, fp) != (size_t)tlen) { fclose(fp); return 0; }

  if (!sn_read_u32(fp, &slen)) { fclose(fp); return 0; }
  if ((size_t)slen > sa_buf_cap) { fclose(fp); return 0; }
  if (fread(sa_buf, sizeof(sa_idx_t), slen, fp) != (size_t)slen) { fclose(fp); return 0; }
  fclose(fp);

  memset(obj, 0, sizeof(*obj));
  obj->text = text_buf;
  obj->text_len = (size_t)tlen;
  obj->sa = sa_buf;
  obj->sa_len = (size_t)slen;
  obj->sa_cap = sa_buf_cap;
  obj->build_flags = build_flags;
  return 1;
}
#endif
