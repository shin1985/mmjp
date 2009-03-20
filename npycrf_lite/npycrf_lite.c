/*
 * npycrf_lite.c
 *
 * UTF-8分かち書き用コンパクトC99ライブラリ実装
 * CRF+言語モデル統合による半マルコフラティス上のビタビデコード
 */

#include "npycrf_lite.h"

#include <string.h>
#include <math.h>

/* ======================================================================
 * 内部ユーティリティ関数
 * ====================================================================== */

/*
 * Q8.8形式同士の乗算
 *
 * (Q8.8 * Q8.8) >> 8 → Q8.8
 * 64ビット中間結果でオーバーフロー回避
 */
static inline npycrf_score_t q16_mul_q8(npycrf_score_t a_q8, npycrf_score_t b_q8) {
  return (npycrf_score_t)(((int64_t)a_q8 * (int64_t)b_q8) >> NPYCRF_Q);
}

/* NULLチェック付き配列アクセス */
static inline uint32_t u32_at(const uint32_t *a, uint32_t i) {
  return a ? a[i] : 0u;
}

/* ======================================================================
 * UTF-8デコード
 * ====================================================================== */

/*
 * UTF-8から1コードポイントをデコード
 *
 * @param s      UTF-8バイト列
 * @param len    バイト長
 * @param io     入出力: 現在位置（成功時1-4進む）
 * @param out_cp 出力: デコードしたコードポイント
 * @return 1=成功, 0=無効なUTF-8
 *
 * 検証内容:
 *  - 継続バイト(0x80-0xBF)の正当性
 *  - オーバーロング表現の拒否
 *  - サロゲートペア(U+D800-U+DFFF)の拒否
 *  - U+10FFFF超の拒否
 */
static int utf8_decode1(const uint8_t *s, size_t len, size_t *io, uint32_t *out_cp) {
  if (!s || !io || !out_cp) return 0;
  size_t i = *io;
  if (i >= len) return 0;

  uint8_t c0 = s[i];

  /* 1バイト文字 (ASCII: 0x00-0x7F) */
  if ((c0 & 0x80u) == 0) {
    *out_cp = (uint32_t)c0;
    *io = i + 1;
    return 1;
  }

  /* 2バイト文字 (0xC0-0xDF で開始) */
  if ((c0 & 0xE0u) == 0xC0u) {
    if (i + 1 >= len) return 0;
    uint8_t c1 = s[i + 1];
    if ((c1 & 0xC0u) != 0x80u) return 0;  /* 継続バイトチェック */
    uint32_t cp = ((uint32_t)(c0 & 0x1Fu) << 6) | (uint32_t)(c1 & 0x3Fu);
    if (cp < 0x80u) return 0;  /* オーバーロング拒否 */
    *out_cp = cp;
    *io = i + 2;
    return 1;
  }

  /* 3バイト文字 (0xE0-0xEF で開始) */
  if ((c0 & 0xF0u) == 0xE0u) {
    if (i + 2 >= len) return 0;
    uint8_t c1 = s[i + 1];
    uint8_t c2 = s[i + 2];
    if ((c1 & 0xC0u) != 0x80u || (c2 & 0xC0u) != 0x80u) return 0;
    uint32_t cp = ((uint32_t)(c0 & 0x0Fu) << 12) |
                  ((uint32_t)(c1 & 0x3Fu) << 6) |
                  (uint32_t)(c2 & 0x3Fu);
    if (cp < 0x800u) return 0;  /* オーバーロング拒否 */
    /* UTF-16サロゲートペアはUTF-8では無効 */
    if (cp >= 0xD800u && cp <= 0xDFFFu) return 0;
    *out_cp = cp;
    *io = i + 3;
    return 1;
  }

  /* 4バイト文字 (0xF0-0xF7 で開始) */
  if ((c0 & 0xF8u) == 0xF0u) {
    if (i + 3 >= len) return 0;
    uint8_t c1 = s[i + 1];
    uint8_t c2 = s[i + 2];
    uint8_t c3 = s[i + 3];
    if ((c1 & 0xC0u) != 0x80u || (c2 & 0xC0u) != 0x80u || (c3 & 0xC0u) != 0x80u) return 0;
    uint32_t cp = ((uint32_t)(c0 & 0x07u) << 18) |
                  ((uint32_t)(c1 & 0x3Fu) << 12) |
                  ((uint32_t)(c2 & 0x3Fu) << 6) |
                  (uint32_t)(c3 & 0x3Fu);
    if (cp < 0x10000u) return 0;   /* オーバーロング拒否 */
    if (cp > 0x10FFFFu) return 0;  /* Unicode範囲外拒否 */
    *out_cp = cp;
    *io = i + 4;
    return 1;
  }

  return 0;  /* 無効な先頭バイト */
}

/*
 * コードポイント開始オフセット配列を構築
 */
size_t npycrf_utf8_make_offsets(const uint8_t *utf8, size_t len, uint16_t *out_off, size_t out_cap) {
  if (!utf8 || !out_off) return 0;
  size_t i = 0;
  size_t n = 0;

  while (i < len) {
    if (n + 1 >= out_cap) return 0;  /* 容量不足 */
    out_off[n++] = (uint16_t)i;
    uint32_t cp = 0;
    if (!utf8_decode1(utf8, len, &i, &cp)) return 0;  /* 無効なUTF-8 */
  }
  if (n >= out_cap) return 0;
  out_off[n] = (uint16_t)len;  /* 終端オフセット */
  return n;
}

/*
 * 境界インデックスをコードポイント単位からバイト単位に変換
 */
void npycrf_boundaries_cp_to_bytes(const uint16_t *cp_off,
                                  const uint16_t *b_cp, size_t b_count,
                                  uint16_t *b_bytes_out) {
  if (!cp_off || !b_cp || !b_bytes_out) return;
  for (size_t i = 0; i < b_count; i++) {
    b_bytes_out[i] = cp_off[b_cp[i]];
  }
}

/* ======================================================================
 * 文字クラス分類
 * ====================================================================== */

/*
 * 粗い文字クラスID
 *
 * MCU向けにテーブルサイズを小さく保つため、
 * 細かい分類は行わない。
 */
enum {
  CC_BOS = NPYCRF_CC_BOS,            /* 文頭（仮想） */
  CC_EOS = NPYCRF_CC_EOS,            /* 文末（仮想） */
  CC_OTHER = NPYCRF_CC_OTHER,        /* その他 */
  CC_SPACE = NPYCRF_CC_SPACE,        /* 空白文字 */
  CC_DIGIT = NPYCRF_CC_DIGIT,        /* 数字 */
  CC_ALPHA = NPYCRF_CC_ALPHA,        /* アルファベット */
  CC_HIRAGANA = NPYCRF_CC_HIRAGANA,  /* ひらがな */
  CC_KATAKANA = NPYCRF_CC_KATAKANA,  /* カタカナ */
  CC_KANJI = NPYCRF_CC_KANJI,        /* 漢字（CJK統合漢字） */
  CC_FULLWIDTH = NPYCRF_CC_FULLWIDTH,/* 全角記号 */
  CC_SYMBOL = NPYCRF_CC_SYMBOL,      /* 記号（ASCII句読点等） */
  CC_UTF8_2BYTE = NPYCRF_CC_UTF8_2BYTE,
  CC_UTF8_3BYTE = NPYCRF_CC_UTF8_3BYTE,
  CC_UTF8_4BYTE = NPYCRF_CC_UTF8_4BYTE,
};

/* Losslessメタ文字 */
#define LOSSLESS_ESCAPE 0x2580u  /* ▀ */
#define LOSSLESS_SPACE  0x2581u  /* ▁ */
#define LOSSLESS_TAB    0x2582u  /* ▂ */
#define LOSSLESS_LF     0x2583u  /* ▃ */
#define LOSSLESS_CR     0x2584u  /* ▄ */

/*
 * ASCII範囲の文字クラス分類（共通処理）
 */
static inline uint8_t char_class_ascii(uint32_t cp) {
  if (cp == ' ' || cp == '\t' || cp == '\n' || cp == '\r') return CC_SPACE;
  if (cp >= '0' && cp <= '9') return CC_DIGIT;
  if ((cp >= 'A' && cp <= 'Z') || (cp >= 'a' && cp <= 'z')) return CC_ALPHA;
  return CC_SYMBOL;
}

/*
 * losslessメタ文字かどうかをチェック
 */
static inline int is_lossless_meta(uint32_t cp) {
  return (cp >= LOSSLESS_ESCAPE && cp <= LOSSLESS_CR);
}

/*
 * UTF-8バイト長からクラスを返す
 */
static inline uint8_t char_class_utf8len(uint32_t cp) {
  if (cp <= 0x7Fu) return char_class_ascii(cp);
  if (cp <= 0x7FFu) return CC_UTF8_2BYTE;
  if (cp <= 0xFFFFu) return CC_UTF8_3BYTE;
  return CC_UTF8_4BYTE;
}

/*
 * ranges配列から二分探索で文字クラスを検索
 */
static uint8_t char_class_from_ranges(const npycrf_cc_range_t *ranges, uint32_t count, uint32_t cp) {
  if (!ranges || count == 0) return CC_OTHER;

  /* 小さい配列は線形探索 */
  if (count <= 8) {
    for (uint32_t i = 0; i < count; i++) {
      if (cp >= ranges[i].lo && cp <= ranges[i].hi) {
        return ranges[i].class_id;
      }
    }
    return CC_OTHER;
  }

  /* 二分探索 */
  uint32_t lo = 0, hi = count;
  while (lo < hi) {
    uint32_t mid = lo + (hi - lo) / 2;
    if (cp < ranges[mid].lo) {
      hi = mid;
    } else if (cp > ranges[mid].hi) {
      lo = mid + 1;
    } else {
      return ranges[mid].class_id;
    }
  }
  return CC_OTHER;
}

/*
 * コードポイントから文字クラスを判定（言語非依存版・公開API）
 */
uint8_t npycrf_char_class_cp(const npycrf_cc_t *cc, uint32_t cp) {
  /* losslessメタ文字は常にSPACE扱い */
  if (is_lossless_meta(cp)) {
    return CC_SPACE;
  }

  /* ASCII範囲は常に従来通り分類 */
  if (cp <= 0x7Fu) {
    return char_class_ascii(cp);
  }

  /* ccがNULLの場合はデフォルト動作（後方互換：日本語ハードコード） */
  if (!cc) {
    /* 日本語文字範囲（後方互換） */
    if (cp >= 0x3040u && cp <= 0x309Fu) return CC_HIRAGANA;
    if (cp >= 0x30A0u && cp <= 0x30FFu) return CC_KATAKANA;
    if (cp >= 0x4E00u && cp <= 0x9FFFu) return CC_KANJI;
    if (cp >= 0xFF00u && cp <= 0xFFEFu) return CC_FULLWIDTH;
    return CC_OTHER;
  }

  /* モードに応じた分類 */
  switch (cc->mode) {
    case NPYCRF_CC_MODE_ASCII:
      return CC_OTHER;

    case NPYCRF_CC_MODE_UTF8LEN:
      return char_class_utf8len(cp);

    case NPYCRF_CC_MODE_RANGES: {
      uint8_t cls = char_class_from_ranges(cc->ranges, cc->range_count, cp);
      if (cls != CC_OTHER) return cls;
      /* フォールバック */
      if (cc->fallback == NPYCRF_CC_MODE_UTF8LEN) {
        return char_class_utf8len(cp);
      }
      return CC_OTHER;
    }

    case NPYCRF_CC_MODE_COMPAT:
      /* 日本語文字範囲（後方互換） */
      if (cp >= 0x3040u && cp <= 0x309Fu) return CC_HIRAGANA;
      if (cp >= 0x30A0u && cp <= 0x30FFu) return CC_KATAKANA;
      if (cp >= 0x4E00u && cp <= 0x9FFFu) return CC_KANJI;
      if (cp >= 0xFF00u && cp <= 0xFFEFu) return CC_FULLWIDTH;
      return CC_OTHER;

    default:
      return CC_OTHER;
  }
}

/*
 * コードポイントから文字クラスを判定（後方互換・内部用）
 *
 * 既存コードとの互換性のため、日本語ハードコード版を残す。
 * 新しいコードでは npycrf_char_class_cp() を使用すること。
 */
static uint8_t char_class(uint32_t cp) {
  return npycrf_char_class_cp(NULL, cp);
}

/* ======================================================================
 * CRF素性ルックアップ
 * ====================================================================== */

/*
 * ソート済みキーテーブルから素性重みを二分探索
 *
 * @param crf  CRFモデル
 * @param key  素性キー（npycrf_feat_key()で生成）
 * @return 重み（Q8.8）、未発見時は0
 */
static int16_t crf_lookup_w(const npycrf_crf_t *crf, uint32_t key) {
  if (!crf || !crf->feat_key || !crf->feat_w || crf->feat_count == 0) return 0;

  uint32_t lo = 0;
  uint32_t hi = crf->feat_count;
  while (lo < hi) {
    uint32_t mid = lo + (hi - lo) / 2u;
    uint32_t k = crf->feat_key[mid];
    if (k == key) return crf->feat_w[mid];
    if (k < key) lo = mid + 1u;
    else hi = mid;
  }
  return 0;  /* 未発見 */
}

/*
 * 位置iでの放射スコアを計算
 *
 * テンプレート0-4の素性重みを合計:
 *  0: 現在文字クラス
 *  1: 前文字クラス
 *  2: 次文字クラス
 *  3: 前・現在ペア
 *  4: 現在・次ペア
 *
 * @param crf    CRFモデル
 * @param label  ラベル（0=内部, 1=開始）
 * @param prev_c 前位置の文字クラス
 * @param cur_c  現在位置の文字クラス
 * @param next_c 次位置の文字クラス
 * @return 放射スコア（Q8.8、int16_t範囲にクランプ）
 */
static int16_t crf_emit_pos(const npycrf_crf_t *crf, uint8_t label,
                            uint8_t prev_c, uint8_t cur_c, uint8_t next_c) {
  int32_t sum = 0;
  sum += crf_lookup_w(crf, npycrf_feat_key(0, label, cur_c, 0));
  sum += crf_lookup_w(crf, npycrf_feat_key(1, label, prev_c, 0));
  sum += crf_lookup_w(crf, npycrf_feat_key(2, label, next_c, 0));
  sum += crf_lookup_w(crf, npycrf_feat_key(3, label, prev_c, cur_c));
  sum += crf_lookup_w(crf, npycrf_feat_key(4, label, cur_c, next_c));

  /* int16_t範囲にクランプ */
  if (sum > 32767) sum = 32767;
  if (sum < -32768) sum = -32768;
  return (int16_t)sum;
}

/* ======================================================================
 * ダブル配列トライ値ヘルパー
 * ====================================================================== */

/*
 * 読み取り専用トライでの遷移（ローカル実装）
 *
 * NOTE: 提供されるダブル配列トライはbase/check配列を公開。
 *       終端ノードには単語IDを負数としてエンコード: base[term] = -(id + 1)
 */
static da_index_t da_next_ro_local(const da_trie_ro_t *da, da_index_t cur, uint8_t code) {
  if (!da || !da->base || !da->check) return 0;
  if (cur <= 0 || (size_t)cur >= da->capacity) return 0;

  da_index_t b = da->base[cur];
  if (b <= 0) return 0;  /* 負のBASEは終端値 */

  size_t idx = (size_t)b + (size_t)code;
  if (idx >= da->capacity) return 0;
  if (da->check[idx] == cur) return (da_index_t)idx;
  return 0;
}

/*
 * 終端値（単語ID）を取得
 */
int npycrf_da_ro_get_term_value(const da_trie_ro_t *da, const uint8_t *key, size_t key_len, npycrf_id_t *out_id) {
  if (!da || !key || !out_id) return 0;

  da_index_t cur = 1;  /* ルートノード */

  /* キーバイトを順に遷移 */
  for (size_t i = 0; i < key_len; i++) {
    cur = da_next_ro_local(da, cur, key[i]);
    if (cur == 0) return 0;  /* 遷移失敗 */
  }

  /* ヌル文字(0)で終端ノードに遷移 */
  da_index_t term = da_next_ro_local(da, cur, 0u);
  if (term == 0) return 0;

  /* BASE値から単語IDを復元: id = -base - 1 */
  da_index_t v = da->base[term];
  if (v >= 0) return 0;  /* 負でなければ終端値なし */

  uint32_t id = (uint32_t)(-v - 1);
  if (id > 0xFFFFu) return 0;  /* 16ビット範囲外 */

  *out_id = (npycrf_id_t)id;
  return 1;
}

/*
 * 終端値（単語ID）を設定
 */
int npycrf_da_set_term_value(da_trie_t *da, const uint8_t *key, size_t key_len, npycrf_id_t id) {
  if (!da || !key) return DA_ERR_BADARG;

  /* まずキーをトライに追加 */
  int rc = da_trie_add_bytes(da, key, key_len);
  if (rc != DA_OK) return rc;

  /* 終端ノードまで遷移 */
  da_index_t cur = 1;
  for (size_t i = 0; i < key_len; i++) {
    da_index_t b = da->base[cur];
    if (b <= 0) return DA_ERR_BADARG;
    size_t idx = (size_t)b + (size_t)key[i];
    if (idx >= da->capacity) return DA_ERR_BADARG;
    if (da->check[idx] != cur) return DA_ERR_BADARG;
    cur = (da_index_t)idx;
  }

  /* ヌル文字終端ノードのBASE値を設定 */
  {
    da_index_t b = da->base[cur];
    if (b <= 0) return DA_ERR_BADARG;
    size_t tidx = (size_t)b + 0u;
    if (tidx >= da->capacity) return DA_ERR_BADARG;
    if (da->check[tidx] != cur) return DA_ERR_BADARG;
    da->base[tidx] = (da_index_t)(-(int32_t)id - 1);
  }
  return DA_OK;
}

/* ======================================================================
 * ワークスペース管理
 * ====================================================================== */

/*
 * 必要なワークバッファサイズを計算
 *
 * メモリレイアウト:
 *  cp_off:      (max_n_cp+1) * uint16_t
 *  emit0/emit1: max_n_cp * int16_t × 2
 *  pref_emit0:  (max_n_cp+1) * int32_t
 *  span_id:     (max_n_cp+1)*(max_word_len+1) * uint16_t
 *  span_luni:   (max_n_cp+1)*(max_word_len+1) * int16_t
 *  bp_prevlen:  (max_n_cp+1)*(max_word_len+1) * uint8_t
 *  dp_ring:     (max_word_len+1)*(max_word_len+1) * int32_t
 */
size_t npycrf_workbuf_size(uint16_t max_n_cp, uint16_t max_word_len) {
  size_t ncp1 = (size_t)max_n_cp + 1u;
  size_t L1 = (size_t)max_word_len + 1u;
  size_t spanN = ncp1 * L1;

  /* アラインメントパディングを考慮 */
  size_t bytes = 0;
  bytes += 2;  /* align(2) for cp_off */
  bytes += ncp1 * sizeof(uint16_t);              /* cp_off */
  bytes += 2;  /* align(2) for emit0 */
  bytes += (size_t)max_n_cp * sizeof(int16_t);   /* emit0 */
  bytes += 2;  /* align(2) for emit1 */
  bytes += (size_t)max_n_cp * sizeof(int16_t);   /* emit1 */
  bytes += 4;  /* align(4) for pref_emit0 */
  bytes += ncp1 * sizeof(int32_t);               /* pref_emit0 */
  bytes += 2;  /* align(2) for span_id */
  bytes += spanN * sizeof(npycrf_id_t);          /* span_id */
  bytes += 2;  /* align(2) for span_luni */
  bytes += spanN * sizeof(int16_t);              /* span_luni */
  bytes += spanN * sizeof(uint8_t);              /* bp_prevlen */
  bytes += 4;  /* align(4) for dp_ring */
  bytes += L1 * L1 * sizeof(npycrf_score_t);     /* dp_ring */
  return bytes;
}

/*
 * ポインタを指定アラインメントに調整
 */
static void *align_ptr(void *p, size_t align) {
  uintptr_t x = (uintptr_t)p;
  uintptr_t m = (uintptr_t)(align - 1u);
  x = (x + m) & ~m;
  return (void *)x;
}

/*
 * ワーク構造体を初期化（バッファを内部で分割）
 */
int npycrf_work_init(npycrf_work_t *w, void *buf, size_t buf_size,
                     uint16_t max_n_cp, uint16_t max_word_len) {
  if (!w || !buf) return -1;
  memset(w, 0, sizeof(*w));
  w->max_n_cp = max_n_cp;
  w->max_word_len = max_word_len;

  size_t need = npycrf_workbuf_size(max_n_cp, max_word_len);
  if (buf_size < need) return -2;  /* バッファ不足 */

  uint8_t *p = (uint8_t *)buf;

  /* 各ブロックを適切なアラインメントで配置 */

  /* cp_off: uint16_t配列 */
  p = (uint8_t *)align_ptr(p, 2);
  w->cp_off = (uint16_t *)p;
  p += ((size_t)max_n_cp + 1u) * sizeof(uint16_t);

  /* emit0: int16_t配列 */
  p = (uint8_t *)align_ptr(p, 2);
  w->emit0 = (int16_t *)p;
  p += (size_t)max_n_cp * sizeof(int16_t);

  /* emit1: int16_t配列 */
  p = (uint8_t *)align_ptr(p, 2);
  w->emit1 = (int16_t *)p;
  p += (size_t)max_n_cp * sizeof(int16_t);

  /* pref_emit0: int32_t配列 */
  p = (uint8_t *)align_ptr(p, 4);
  w->pref_emit0 = (int32_t *)p;
  p += ((size_t)max_n_cp + 1u) * sizeof(int32_t);

  size_t ncp1 = (size_t)max_n_cp + 1u;
  size_t L1 = (size_t)max_word_len + 1u;
  size_t spanN = ncp1 * L1;

  /* span_id: uint16_t配列 */
  p = (uint8_t *)align_ptr(p, 2);
  w->span_id = (npycrf_id_t *)p;
  p += spanN * sizeof(npycrf_id_t);

  /* span_luni: int16_t配列 */
  p = (uint8_t *)align_ptr(p, 2);
  w->span_luni = (int16_t *)p;
  p += spanN * sizeof(int16_t);

  /* bp_prevlen: uint8_t配列（アラインメント不要） */
  p = (uint8_t *)align_ptr(p, 1);
  w->bp_prevlen = (uint8_t *)p;
  p += spanN * sizeof(uint8_t);

  /* dp_ring: int32_t配列 */
  p = (uint8_t *)align_ptr(p, 4);
  w->dp_ring = (npycrf_score_t *)p;
  p += L1 * L1 * sizeof(npycrf_score_t);

  (void)p;  /* 未使用警告抑制 */
  return 0;
}

/* ======================================================================
 * 言語モデルヘルパー
 * ====================================================================== */

/*
 * ユニグラム対数確率を取得
 *
 * @param lm      言語モデル
 * @param id      単語ID
 * @param len_cp  単語長（コードポイント数、未知語ペナルティ計算用）
 * @return 対数確率（Q8.8）
 *
 * 未知語の場合: unk_base + unk_per_cp * len_cp
 */
static int16_t lm_unigram_logp(const npycrf_lm_t *lm, npycrf_id_t id, uint16_t len_cp) {
  if (!lm) return 0;

  /* 既知語の場合はテーブルから取得 */
  if (id != NPYCRF_ID_NONE && id != NPYCRF_ID_BOS &&
      (uint32_t)id < lm->vocab_size && lm->logp_uni) {
    return lm->logp_uni[id];
  }

  /* 未知語ペナルティを計算 */
  int32_t v = (int32_t)lm->unk_base + (int32_t)lm->unk_per_cp * (int32_t)len_cp;
  if (v > 32767) v = 32767;
  if (v < -32768) v = -32768;
  return (int16_t)v;
}

/*
 * バイグラム対数確率を取得（バックオフ付き）
 *
 * @param lm           言語モデル
 * @param prev         前単語ID
 * @param curr         現単語ID
 * @param curr_backoff バックオフ値（ユニグラム確率）
 * @return 対数確率（Q8.8）
 *
 * バイグラムテーブルにない場合はcurr_backoffを返す
 */
static int16_t lm_bigram_logp(const npycrf_lm_t *lm, npycrf_id_t prev, npycrf_id_t curr, int16_t curr_backoff) {
  if (!lm) return curr_backoff;
  if (!lm->bigram_key || !lm->logp_bi || lm->bigram_size == 0) return curr_backoff;
  if (prev == NPYCRF_ID_NONE || curr == NPYCRF_ID_NONE) return curr_backoff;

  /* キー構築: (prev_id << 16) | curr_id */
  uint32_t key = ((uint32_t)prev << 16) | (uint32_t)curr;

  /* 二分探索 */
  uint32_t lo = 0;
  uint32_t hi = lm->bigram_size;
  while (lo < hi) {
    uint32_t mid = lo + (hi - lo) / 2u;
    uint32_t k = lm->bigram_key[mid];
    if (k == key) return lm->logp_bi[mid];
    if (k < key) lo = mid + 1u;
    else hi = mid;
  }
  return curr_backoff;  /* 未発見→バックオフ */
}

/*
 * スパンテーブルインデックス計算
 *
 * 行優先: index = end_pos * (L+1) + len
 */
static inline size_t span_index(uint16_t end_pos, uint16_t len, uint16_t L) {
  return (size_t)end_pos * ((size_t)L + 1u) + (size_t)len;
}

/* ======================================================================
 * 事前計算
 * ====================================================================== */

/*
 * CRF放射スコアを事前計算
 *
 * 各位置でラベル0/1の放射スコアを計算し、
 * emit0の累積和も計算（区間和の高速計算用）
 */
static int precompute_emissions(const npycrf_model_t *m,
                                const uint8_t *utf8, size_t len,
                                const uint16_t *off, uint16_t n_cp,
                                npycrf_work_t *w) {
  if (!m || !utf8 || !off || !w) return -1;

  /* 各位置の文字クラスと放射スコアを計算 */
  for (uint16_t i = 0; i < n_cp; i++) {
    size_t pos = off[i];
    size_t end = off[i + 1];
    (void)end;

    /* 現在位置のコードポイントをデコード */
    uint32_t cp = 0;
    size_t tmp = pos;
    if (!utf8_decode1(utf8, len, &tmp, &cp)) return -2;

    uint8_t cur = char_class(cp);

    /* 前位置の文字クラス（先頭ならBOS） */
    uint8_t prev = CC_BOS;
    if (i > 0) {
      size_t ppos = off[i - 1];
      uint32_t pcp = 0;
      size_t t2 = ppos;
      if (!utf8_decode1(utf8, len, &t2, &pcp)) return -2;
      prev = char_class(pcp);
    }

    /* 次位置の文字クラス（末尾ならEOS） */
    uint8_t next = CC_EOS;
    if (i + 1 < n_cp) {
      size_t npos = off[i + 1];
      uint32_t ncpv = 0;
      size_t t3 = npos;
      if (!utf8_decode1(utf8, len, &t3, &ncpv)) return -2;
      next = char_class(ncpv);
    }

    /* ラベル0/1の放射スコアを計算 */
    w->emit0[i] = crf_emit_pos(&m->crf, 0u, prev, cur, next);
    w->emit1[i] = crf_emit_pos(&m->crf, 1u, prev, cur, next);
  }

  /* emit0の累積和を計算（区間[s+1, t)の和 = pref[t] - pref[s+1]） */
  w->pref_emit0[0] = 0;
  for (uint16_t i = 0; i < n_cp; i++) {
    w->pref_emit0[i + 1] = (int32_t)w->pref_emit0[i] + (int32_t)w->emit0[i];
  }

  return 0;
}

/*
 * スパン情報を事前計算
 *
 * 全(終了位置, 長さ)ペアについて:
 *  - 辞書トライから単語IDを検索
 *  - ユニグラム/未知語対数確率を計算
 */
static int precompute_spans(const npycrf_model_t *m,
                            const uint8_t *utf8, size_t len,
                            const uint16_t *off, uint16_t n_cp,
                            npycrf_work_t *w) {
  if (!m || !utf8 || !off || !w) return -1;
  uint16_t L = m->max_word_len;
  if (L == 0) return -1;

  if (n_cp > w->max_n_cp || L > w->max_word_len) return -2;

  size_t ncp1 = (size_t)n_cp + 1u;
  size_t L1 = (size_t)L + 1u;
  size_t spanN = ncp1 * L1;

  /* スパンテーブルを初期化 */
  for (size_t i = 0; i < spanN; i++) {
    w->span_id[i] = NPYCRF_ID_NONE;
    w->span_luni[i] = 0;
    w->bp_prevlen[i] = 0;
  }

  /* BOS状態（位置0、長さ0）を設定 */
  w->span_id[span_index(0, 0, L)] = NPYCRF_ID_BOS;
  w->span_luni[span_index(0, 0, L)] = 0;

  /* 各開始位置からトライを辿って既知語IDを設定 */
  for (uint16_t start_cp = 0; start_cp < n_cp; start_cp++) {
    da_index_t node = 1;  /* トライルート */
    uint16_t max_l = (uint16_t)((start_cp + L <= n_cp) ? L : (n_cp - start_cp));

    for (uint16_t l = 1; l <= max_l; l++) {
      /* start_cp + l - 1 位置のコードポイントのバイトを消費 */
      uint16_t cp_i = (uint16_t)(start_cp + l - 1);
      size_t b0 = off[cp_i];
      size_t b1 = off[cp_i + 1];
      for (size_t bi = b0; bi < b1; bi++) {
        node = da_next_ro_local(&m->lm.trie, node, utf8[bi]);
        if (node == 0) break;  /* 遷移失敗 */
      }
      if (node == 0) break;

      /* ヌル文字で終端チェック */
      da_index_t term = da_next_ro_local(&m->lm.trie, node, 0u);
      if (term != 0) {
        da_index_t v = m->lm.trie.base[term];
        if (v < 0) {
          uint32_t id = (uint32_t)(-v - 1);
          if (id <= 0xFFFFu) {
            uint16_t end_cp = (uint16_t)(start_cp + l);
            w->span_id[span_index(end_cp, l, L)] = (npycrf_id_t)id;
          }
        }
      }
    }
  }

  /* 全スパン(長さ>=1)のユニグラム/OOV対数確率を計算 */
  for (uint16_t end_cp = 1; end_cp <= n_cp; end_cp++) {
    uint16_t max_l = (uint16_t)((end_cp <= L) ? end_cp : L);
    for (uint16_t l = 1; l <= max_l; l++) {
      size_t idx = span_index(end_cp, l, L);
      npycrf_id_t id = w->span_id[idx];
      w->span_luni[idx] = lm_unigram_logp(&m->lm, id, l);
    }
  }

  return 0;
}

/* ======================================================================
 * CRFセグメントスコア
 * ====================================================================== */

/*
 * 単語スパン[s, t)のCRFスコアを計算
 *
 * ラベル系列: 1, 0, 0, ..., 0 (長さkの単語)
 * 次の単語境界でラベル1に遷移
 *
 * スコア計算:
 *  - emit1[s] (開始位置)
 *  - trans10 (開始→内部)
 *  - Σemit0[s+1..t-1] (内部位置)
 *  - (k-2) * trans00 (内部→内部の繰り返し)
 *  - trans01 (内部→次単語開始)
 */
static inline npycrf_score_t crf_seg_score(const npycrf_model_t *m, const npycrf_work_t *w,
                                          uint16_t s, uint16_t t) {
  uint16_t k = (uint16_t)(t - s);
  if (k == 0) return 0;

  /* 1文字単語: ラベル1のみ、次もラベル1 */
  if (k == 1) {
    return (npycrf_score_t)w->emit1[s] + (npycrf_score_t)m->crf.trans11;
  }

  /* 2文字以上: 1,0,...,0 → 1 */
  npycrf_score_t score = 0;
  score += (npycrf_score_t)w->emit1[s];      /* 開始位置の放射 */
  score += (npycrf_score_t)m->crf.trans10;   /* 1→0 遷移 */

  /* emit0[s+1..t-1]の区間和（累積和テーブルから計算） */
  int32_t sum0 = w->pref_emit0[t] - w->pref_emit0[s + 1u];
  score += (npycrf_score_t)sum0;

  /* 0→0 遷移を (k-2) 回 */
  score += (npycrf_score_t)((int32_t)m->crf.trans00 * (int32_t)(k - 2u));

  /* 最後に 0→1 遷移（次単語の境界） */
  score += (npycrf_score_t)m->crf.trans01;

  return score;
}

/* ======================================================================
 * ビタビデコード
 * ====================================================================== */

/*
 * 半マルコフラティス上のビタビアルゴリズム
 *
 * 状態: (位置, 最後の単語長)
 * 遷移: 長さkの単語を追加して位置を進める
 *
 * DPリングバッファ:
 *  - メモリ効率のため、位置を (L+1) でmod
 *  - 過去L+1位置分のスコアのみ保持
 *
 * バックポインタ:
 *  - 各(位置, 長さ)での最適な前単語長を記録
 *  - 最後にバックトラックして境界を復元
 */
int npycrf_decode(const npycrf_model_t *model,
                  const uint8_t *utf8, size_t len,
                  npycrf_work_t *work,
                  uint16_t *out_b_cp, size_t out_b_cap,
                  size_t *out_b_count,
                  npycrf_score_t *out_best_score) {
  if (!model || !utf8 || !work || !out_b_cp || !out_b_count) return -1;
  if (model->max_word_len == 0) return -1;

  /* 1) コードポイントオフセットを構築 */
  if (!work->cp_off || work->max_n_cp == 0) return -2;
  size_t n_cp_sz = npycrf_utf8_make_offsets(utf8, len, work->cp_off, (size_t)work->max_n_cp + 1u);
  if (n_cp_sz == 0) return -3;
  if (n_cp_sz > work->max_n_cp) return -3;

  uint16_t n_cp = (uint16_t)n_cp_sz;
  uint16_t L = model->max_word_len;
  if (L > work->max_word_len) return -4;

  /* 最低2境界スロット必要（0とn） */
  if (out_b_cap < 2u) return -5;

  /* 2) 放射スコアとスパン情報を事前計算 */
  int rc = precompute_emissions(model, utf8, len, work->cp_off, n_cp, work);
  if (rc != 0) return -10;
  rc = precompute_spans(model, utf8, len, work->cp_off, n_cp, work);
  if (rc != 0) return -11;

  /* 3) 半マルコフラティス上のビタビDP */
  size_t L1 = (size_t)L + 1u;

  /* DPリングバッファを負の無限大で初期化 */
  for (size_t i = 0; i < L1 * L1; i++) work->dp_ring[i] = NPYCRF_SCORE_NEG_INF;

  /* 初期状態: dp[0][0] = bos_to1 */
  work->dp_ring[0 * L1 + 0] = (npycrf_score_t)model->crf.bos_to1;

  /* 前向きDP */
  for (uint16_t pos = 1; pos <= n_cp; pos++) {
    /* この位置のリング行をクリア */
    uint16_t row = (uint16_t)(pos % (L + 1u));
    for (uint16_t k = 0; k <= L; k++) {
      work->dp_ring[(size_t)row * L1 + k] = NPYCRF_SCORE_NEG_INF;
    }

    uint16_t kmax = (uint16_t)((pos <= L) ? pos : L);

    /* 長さkの単語で位置posに到達 */
    for (uint16_t k = 1; k <= kmax; k++) {
      uint16_t start = (uint16_t)(pos - k);

      /* CRFセグメントスコア */
      npycrf_score_t seg = crf_seg_score(model, work, start, pos);

      /* 現在スパンの情報 */
      size_t idx_curr = span_index(pos, k, L);
      npycrf_id_t curr_id = work->span_id[idx_curr];
      int16_t curr_luni = work->span_luni[idx_curr];

      /* 前状態からの最良スコアを探索 */
      npycrf_score_t best = NPYCRF_SCORE_NEG_INF;
      uint8_t best_j = 0;

      uint16_t prev_pos = start;
      uint16_t prev_row = (uint16_t)(prev_pos % (L + 1u));

      /* j=0（BOS）は prev_pos==0 の場合のみ有効 */
      if (prev_pos == 0) {
        npycrf_score_t prev_score = work->dp_ring[(size_t)prev_row * L1 + 0];
        if (prev_score != NPYCRF_SCORE_NEG_INF) {
          int16_t lm = lm_bigram_logp(&model->lm, NPYCRF_ID_BOS, curr_id, curr_luni);
          npycrf_score_t add = q16_mul_q8((npycrf_score_t)model->lambda0, (npycrf_score_t)lm);
          npycrf_score_t cand = prev_score + seg + add;
          best = cand;
          best_j = 0;
        }
      }

      /* j>=1: 前単語の長さ */
      uint16_t jmax = (uint16_t)((prev_pos <= L) ? prev_pos : L);
      for (uint16_t j = 1; j <= jmax; j++) {
        npycrf_score_t prev_score = work->dp_ring[(size_t)prev_row * L1 + j];
        if (prev_score == NPYCRF_SCORE_NEG_INF) continue;

        size_t idx_prev = span_index(prev_pos, j, L);
        npycrf_id_t prev_id = work->span_id[idx_prev];

        /* バイグラムLMスコア */
        int16_t lm = lm_bigram_logp(&model->lm, prev_id, curr_id, curr_luni);
        npycrf_score_t add = q16_mul_q8((npycrf_score_t)model->lambda0, (npycrf_score_t)lm);

        npycrf_score_t cand = prev_score + seg + add;
        if (cand > best) {
          best = cand;
          best_j = (uint8_t)j;
        }
      }

      work->dp_ring[(size_t)row * L1 + k] = best;
      work->bp_prevlen[span_index(pos, k, L)] = best_j;
    }
  }

  /* 4) 最良終端状態を選択 */
  uint16_t end_row = (uint16_t)(n_cp % (L + 1u));
  npycrf_score_t best_final = NPYCRF_SCORE_NEG_INF;
  uint16_t best_k = 0;

  uint16_t kmax_end = (uint16_t)((n_cp <= L) ? n_cp : L);
  for (uint16_t k = 1; k <= kmax_end; k++) {
    npycrf_score_t v = work->dp_ring[(size_t)end_row * L1 + k];
    if (v > best_final) {
      best_final = v;
      best_k = k;
    }
  }

  /* 空文字列の場合 */
  if (n_cp == 0) {
    out_b_cp[0] = 0;
    out_b_cp[1] = 0;
    *out_b_count = 2;
    if (out_best_score) *out_best_score = work->dp_ring[0];
    return 0;
  }

  if (best_k == 0 || best_final == NPYCRF_SCORE_NEG_INF) return -20;

  /* 5) バックトラックで境界を復元 */
  if (out_b_cap < (size_t)n_cp + 1u) {
    return -21;  /* 出力バッファ不足 */
  }

  /* 逆順で収集（n, ..., 0） */
  size_t bcnt = 0;
  uint16_t pos = n_cp;
  uint16_t k = best_k;
  while (1) {
    out_b_cp[bcnt++] = pos;
    if (pos == 0) break;

    uint16_t start = (uint16_t)(pos - k);
    uint8_t j = work->bp_prevlen[span_index(pos, k, L)];
    pos = start;
    k = (uint16_t)j;
    if (pos == 0) {
      out_b_cp[bcnt++] = 0;
      break;
    }
    if (k == 0) {
      return -22;  /* 無効なバックポインタ */
    }
    if (bcnt > (size_t)n_cp + 1u) return -23;
  }

  /* 境界を正順に反転 */
  for (size_t i = 0; i < bcnt / 2u; i++) {
    uint16_t tmp = out_b_cp[i];
    out_b_cp[i] = out_b_cp[bcnt - 1u - i];
    out_b_cp[bcnt - 1u - i] = tmp;
  }

  /* 境界が0で始まりn_cpで終わることを確認 */
  if (out_b_cp[0] != 0 || out_b_cp[bcnt - 1u] != n_cp) return -24;

  *out_b_count = bcnt;
  if (out_best_score) *out_best_score = best_final;
  return 0;
}

/* ======================================================================
 * Subword Regularization（確率的分割）
 * ====================================================================== */

static inline uintptr_t align_up_uintptr(uintptr_t p, size_t align) {
  return (p + (uintptr_t)(align - 1u)) & ~(uintptr_t)(align - 1u);
}

static inline uint32_t xs32(uint32_t *state) {
  uint32_t x = (state && *state) ? *state : 0x12345678u;
  x ^= x << 13;
  x ^= x >> 17;
  x ^= x << 5;
  if (state) *state = x;
  return x;
}

static inline double xs32_u01(uint32_t *state) {
  /* 24-bit mantissa -> [0,1) */
  return (double)(xs32(state) >> 8) * (1.0 / 16777216.0);
}

static inline double logsumexp2(double a, double b) {
  if (a == -INFINITY) return b;
  if (b == -INFINITY) return a;
  double m = (a > b) ? a : b;
  return m + log(exp(a - m) + exp(b - m));
}

static inline double score_q88_to_f(npycrf_score_t q) {
  return (double)q / (double)NPYCRF_SCORE_SCALE;
}

size_t npycrf_samplebuf_size(uint16_t max_n_cp, uint16_t max_word_len) {
  size_t states = ((size_t)max_n_cp + 1u) * ((size_t)max_word_len + 1u);
  /* +16 for alignment slack */
  return states * sizeof(double) + 16u;
}

/*
 * Forward-Filtering Backward-Sampling (FFBS)
 *  - forward: semi-markov lattice log-sum DP
 *  - backward: sample previous state using alpha and local edge scores
 */
int npycrf_decode_sample(const npycrf_model_t *model,
                         const uint8_t *utf8, size_t len,
                         npycrf_work_t *work,
                         void *sample_buf, size_t sample_buf_size,
                         double temperature,
                         uint32_t seed,
                         uint16_t *out_b_cp, size_t out_b_cap,
                         size_t *out_b_count,
                         npycrf_score_t *out_sample_score) {
  if (!model || !utf8 || !work || !sample_buf || sample_buf_size == 0 || !out_b_cp || !out_b_count) return -1;
  if (model->max_word_len == 0) return -1;
  if (!(temperature > 0.0) || isnan(temperature) || isinf(temperature)) temperature = 1.0;

  /* 1) offsets */
  if (!work->cp_off || work->max_n_cp == 0) return -2;
  size_t n_cp_sz = npycrf_utf8_make_offsets(utf8, len, work->cp_off, (size_t)work->max_n_cp + 1u);
  if (n_cp_sz == 0) return -3;
  if (n_cp_sz > work->max_n_cp) return -3;
  uint16_t n_cp = (uint16_t)n_cp_sz;
  uint16_t L = model->max_word_len;
  if (L > work->max_word_len) return -4;
  if (out_b_cap < 2u) return -5;

  /* 2) precompute */
  int rc = precompute_emissions(model, utf8, len, work->cp_off, n_cp, work);
  if (rc != 0) return -10;
  rc = precompute_spans(model, utf8, len, work->cp_off, n_cp, work);
  if (rc != 0) return -11;

  /* 3) parse sample buffer -> alpha table */
  size_t L1 = (size_t)L + 1u;
  size_t states = ((size_t)n_cp + 1u) * L1;
  uintptr_t p0 = (uintptr_t)sample_buf;
  uintptr_t p1 = align_up_uintptr(p0, (size_t)sizeof(double));
  size_t pad = (size_t)(p1 - p0);
  size_t need = pad + states * sizeof(double);
  if (sample_buf_size < need) return -12;
  double *alpha = (double *)p1;

  /* init alpha with -inf */
  for (size_t i = 0; i < states; i++) alpha[i] = -INFINITY;

  /* alpha[0,0] = bos_to1 */
  alpha[0] = score_q88_to_f((npycrf_score_t)model->crf.bos_to1) / temperature;

  /* forward DP */
  for (uint16_t pos = 1; pos <= n_cp; pos++) {
    uint16_t kmax = (uint16_t)((pos <= L) ? pos : L);
    for (uint16_t k = 1; k <= kmax; k++) {
      uint16_t start = (uint16_t)(pos - k);

      npycrf_score_t seg = crf_seg_score(model, work, start, pos);
      size_t idx_curr = span_index(pos, k, L);
      npycrf_id_t curr_id = work->span_id[idx_curr];
      int16_t curr_luni = work->span_luni[idx_curr];

      double log_sum = -INFINITY;

      /* j=0 only if start==0 */
      if (start == 0) {
        double prev = alpha[0 * L1 + 0];
        if (prev != -INFINITY) {
          int16_t lm = lm_bigram_logp(&model->lm, NPYCRF_ID_BOS, curr_id, curr_luni);
          npycrf_score_t add = q16_mul_q8((npycrf_score_t)model->lambda0, (npycrf_score_t)lm);
          double edge = (score_q88_to_f(seg + add) / temperature);
          log_sum = prev + edge;
        }
      } else {
        uint16_t jmax = (uint16_t)((start <= L) ? start : L);
        for (uint16_t j = 1; j <= jmax; j++) {
          double prev = alpha[(size_t)start * L1 + (size_t)j];
          if (prev == -INFINITY) continue;
          size_t idx_prev = span_index(start, j, L);
          npycrf_id_t prev_id = work->span_id[idx_prev];

          int16_t lm = lm_bigram_logp(&model->lm, prev_id, curr_id, curr_luni);
          npycrf_score_t add = q16_mul_q8((npycrf_score_t)model->lambda0, (npycrf_score_t)lm);
          double edge = (score_q88_to_f(seg + add) / temperature);
          log_sum = logsumexp2(log_sum, prev + edge);
        }
      }

      alpha[(size_t)pos * L1 + (size_t)k] = log_sum;
    }
  }

  /* empty string */
  if (n_cp == 0) {
    out_b_cp[0] = 0;
    out_b_cp[1] = 0;
    *out_b_count = 2;
    if (out_sample_score) *out_sample_score = (npycrf_score_t)model->crf.bos_to1;
    return 0;
  }

  /* 4) sample final k */
  uint16_t kmax_end = (uint16_t)((n_cp <= L) ? n_cp : L);
  double logZ = -INFINITY;
  for (uint16_t k = 1; k <= kmax_end; k++) {
    logZ = logsumexp2(logZ, alpha[(size_t)n_cp * L1 + (size_t)k]);
  }
  if (logZ == -INFINITY) return -20;

  double u = xs32_u01(&seed);
  double cdf = 0.0;
  uint16_t cur_k = 1;
  for (uint16_t k = 1; k <= kmax_end; k++) {
    double lp = alpha[(size_t)n_cp * L1 + (size_t)k] - logZ;
    double p = exp(lp);
    cdf += p;
    if (u <= cdf) {
      cur_k = k;
      break;
    }
  }

  /* 5) backward sample boundaries */
  if (out_b_cap < (size_t)n_cp + 1u) return -21;

  size_t bcnt = 0;
  uint16_t pos = n_cp;
  uint16_t k = cur_k;
  while (1) {
    out_b_cp[bcnt++] = pos;
    if (pos == 0) break;
    uint16_t start = (uint16_t)(pos - k);
    if (start == 0) {
      out_b_cp[bcnt++] = 0;
      break;
    }

    /* sample prev len j given current (pos,k)
     *   p(j | pos,k) ∝ exp(alpha[start,j] + edge(j→k))
     *   where edge includes CRF segment score + LM bigram
     */
    uint16_t jmax = (uint16_t)((start <= L) ? start : L);

    npycrf_score_t seg = crf_seg_score(model, work, start, pos);
    size_t idx_curr = span_index(pos, k, L);
    npycrf_id_t curr_id = work->span_id[idx_curr];
    int16_t curr_luni = work->span_luni[idx_curr];

    double alpha_cur = alpha[(size_t)pos * L1 + (size_t)k];

    /* 1st pass: find max log-weight */
    double maxlw = -INFINITY;
    size_t valid = 0;
    for (uint16_t j = 1; j <= jmax; j++) {
      double a_prev = alpha[(size_t)start * L1 + (size_t)j];
      if (a_prev == -INFINITY) continue;
      size_t idx_prev = span_index(start, j, L);
      npycrf_id_t prev_id = work->span_id[idx_prev];
      int16_t lm = lm_bigram_logp(&model->lm, prev_id, curr_id, curr_luni);
      npycrf_score_t add = q16_mul_q8((npycrf_score_t)model->lambda0, (npycrf_score_t)lm);
      double edge = score_q88_to_f(seg + add) / temperature;
      double lw = (a_prev + edge) - alpha_cur;
      if (lw > maxlw) maxlw = lw;
      valid++;
    }
    if (valid == 0 || maxlw == -INFINITY) return -22;

    /* 2nd pass: sum exp */
    double sum = 0.0;
    for (uint16_t j = 1; j <= jmax; j++) {
      double a_prev = alpha[(size_t)start * L1 + (size_t)j];
      if (a_prev == -INFINITY) continue;
      size_t idx_prev = span_index(start, j, L);
      npycrf_id_t prev_id = work->span_id[idx_prev];
      int16_t lm = lm_bigram_logp(&model->lm, prev_id, curr_id, curr_luni);
      npycrf_score_t add = q16_mul_q8((npycrf_score_t)model->lambda0, (npycrf_score_t)lm);
      double edge = score_q88_to_f(seg + add) / temperature;
      double lw = (a_prev + edge) - alpha_cur;
      sum += exp(lw - maxlw);
    }
    if (!(sum > 0.0) || isnan(sum) || isinf(sum)) return -22;

    double r = xs32_u01(&seed) * sum;
    double acc = 0.0;
    uint16_t pick = 1;
    for (uint16_t j = 1; j <= jmax; j++) {
      double a_prev = alpha[(size_t)start * L1 + (size_t)j];
      if (a_prev == -INFINITY) continue;
      size_t idx_prev = span_index(start, j, L);
      npycrf_id_t prev_id = work->span_id[idx_prev];
      int16_t lm = lm_bigram_logp(&model->lm, prev_id, curr_id, curr_luni);
      npycrf_score_t add = q16_mul_q8((npycrf_score_t)model->lambda0, (npycrf_score_t)lm);
      double edge = score_q88_to_f(seg + add) / temperature;
      double lw = (a_prev + edge) - alpha_cur;
      acc += exp(lw - maxlw);
      pick = j;
      if (r <= acc) break;
    }

    pos = start;
    k = pick;
    if (bcnt > (size_t)n_cp + 1u) return -23;
  }

  /* reverse boundaries */
  for (size_t i = 0; i < bcnt / 2u; i++) {
    uint16_t tmp = out_b_cp[i];
    out_b_cp[i] = out_b_cp[bcnt - 1u - i];
    out_b_cp[bcnt - 1u - i] = tmp;
  }
  if (out_b_cp[0] != 0 || out_b_cp[bcnt - 1u] != n_cp) return -24;

  *out_b_count = bcnt;

  /* compute sampled path score in Q8.8 (optional) */
  if (out_sample_score) {
    npycrf_score_t total = (npycrf_score_t)model->crf.bos_to1;
    for (size_t i = 0; i + 1 < bcnt; i++) {
      uint16_t s = out_b_cp[i];
      uint16_t t = out_b_cp[i + 1];
      uint16_t len_cp = (uint16_t)(t - s);
      if (len_cp == 0 || len_cp > L) continue;

      npycrf_score_t seg = crf_seg_score(model, work, s, t);
      size_t idx = span_index(t, len_cp, L);
      npycrf_id_t curr_id = work->span_id[idx];
      int16_t curr_luni = work->span_luni[idx];

      npycrf_id_t prev_id = NPYCRF_ID_BOS;
      if (i > 0) {
        uint16_t ps = out_b_cp[i - 1];
        uint16_t pt = out_b_cp[i];
        uint16_t plen = (uint16_t)(pt - ps);
        if (plen > 0 && plen <= L) {
          prev_id = work->span_id[span_index(pt, plen, L)];
        }
      }

      int16_t lm = lm_bigram_logp(&model->lm, prev_id, curr_id, curr_luni);
      npycrf_score_t add = q16_mul_q8((npycrf_score_t)model->lambda0, (npycrf_score_t)lm);
      total += seg + add;
    }
    *out_sample_score = total;
  }
  return 0;
}

size_t npycrf_nbestbuf_size(uint16_t max_n_cp, uint16_t max_word_len, uint16_t nbest) {
  if (nbest == 0) return 0;
  size_t states = ((size_t)max_n_cp + 1u) * ((size_t)max_word_len + 1u);
  size_t n = (size_t)nbest;
  size_t bytes = 0;
  bytes += 16u; /* alignment slack */
  bytes += states * n * sizeof(npycrf_score_t);
  bytes += states * n * sizeof(uint8_t); /* prevlen */
  bytes += states * n * sizeof(uint8_t); /* prevrank */
  return bytes;
}

int npycrf_decode_nbest(const npycrf_model_t *model,
                        const uint8_t *utf8, size_t len,
                        npycrf_work_t *work,
                        void *nbest_buf, size_t nbest_buf_size,
                        uint16_t nbest,
                        uint16_t *out_b_cp_flat, size_t out_b_cap,
                        size_t *out_b_count,
                        npycrf_score_t *out_scores) {
  if (!model || !utf8 || !work || !nbest_buf || nbest_buf_size == 0 || !out_b_cp_flat || !out_b_count) return -1;
  if (model->max_word_len == 0 || nbest == 0) return -1;

  /* 1) offsets */
  if (!work->cp_off || work->max_n_cp == 0) return -2;
  size_t n_cp_sz = npycrf_utf8_make_offsets(utf8, len, work->cp_off, (size_t)work->max_n_cp + 1u);
  if (n_cp_sz == 0) return -3;
  if (n_cp_sz > work->max_n_cp) return -3;
  uint16_t n_cp = (uint16_t)n_cp_sz;
  uint16_t L = model->max_word_len;
  if (L > work->max_word_len) return -4;
  if (out_b_cap < (size_t)n_cp + 1u) return -5;

  /* 2) precompute */
  int rc = precompute_emissions(model, utf8, len, work->cp_off, n_cp, work);
  if (rc != 0) return -10;
  rc = precompute_spans(model, utf8, len, work->cp_off, n_cp, work);
  if (rc != 0) return -11;

  /* 3) parse buffer */
  size_t L1 = (size_t)L + 1u;
  size_t states = ((size_t)n_cp + 1u) * L1;

  uintptr_t p0 = (uintptr_t)nbest_buf;
  uintptr_t p1 = align_up_uintptr(p0, 4u);
  size_t pad = (size_t)(p1 - p0);
  size_t need_scores = states * (size_t)nbest * sizeof(npycrf_score_t);
  size_t need_prevlen = states * (size_t)nbest * sizeof(uint8_t);
  size_t need_prevrank = states * (size_t)nbest * sizeof(uint8_t);
  size_t need = pad + need_scores + need_prevlen + need_prevrank;
  if (nbest_buf_size < need) return -12;

  uint8_t *p = (uint8_t *)p1;
  npycrf_score_t *dp = (npycrf_score_t *)p;
  p += need_scores;
  uint8_t *bp_len = p;
  p += need_prevlen;
  uint8_t *bp_rank = p;

  /* init */
  size_t total = states * (size_t)nbest;
  for (size_t i = 0; i < total; i++) {
    dp[i] = NPYCRF_SCORE_NEG_INF;
    bp_len[i] = 0;
    bp_rank[i] = 0;
  }

  /* start state (0,0) rank0 */
  dp[0 * (size_t)nbest + 0] = (npycrf_score_t)model->crf.bos_to1;

  /* forward k-best */
  for (uint16_t pos = 1; pos <= n_cp; pos++) {
    uint16_t kmax = (uint16_t)((pos <= L) ? pos : L);
    for (uint16_t k = 1; k <= kmax; k++) {
      uint16_t start = (uint16_t)(pos - k);

      npycrf_score_t seg = crf_seg_score(model, work, start, pos);
      size_t idx_curr = span_index(pos, k, L);
      npycrf_id_t curr_id = work->span_id[idx_curr];
      int16_t curr_luni = work->span_luni[idx_curr];

      /* topK arrays (local) */
      /* NOTE: nbest is runtime; allocate on heap? -> use fixed upper bound via malloc-free insert */
      /* We implement insertion directly into dp for current state via temp arrays allocated with malloc? */
      /* To keep pure C99 without VLA, cap nbest to 64. */
      if (nbest > 64u) return -13;
      npycrf_score_t best_s[64];
      uint8_t best_pl[64];
      uint8_t best_pr[64];
      for (uint16_t r = 0; r < nbest; r++) {
        best_s[r] = NPYCRF_SCORE_NEG_INF;
        best_pl[r] = 0;
        best_pr[r] = 0;
      }

      if (start == 0) {
        /* only prev (0,0) */
        npycrf_score_t add0 = 0;
        int16_t lm = lm_bigram_logp(&model->lm, NPYCRF_ID_BOS, curr_id, curr_luni);
        add0 = q16_mul_q8((npycrf_score_t)model->lambda0, (npycrf_score_t)lm);
        npycrf_score_t edge = seg + add0;
        for (uint16_t pr = 0; pr < nbest; pr++) {
          npycrf_score_t prev = dp[0 * (size_t)nbest + pr];
          if (prev == NPYCRF_SCORE_NEG_INF) continue;
          npycrf_score_t cand = prev + edge;
          /* insert cand into best_s */
          for (uint16_t t = 0; t < nbest; t++) {
            if (cand > best_s[t]) {
              for (uint16_t u = nbest - 1; u > t; u--) {
                best_s[u] = best_s[u - 1];
                best_pl[u] = best_pl[u - 1];
                best_pr[u] = best_pr[u - 1];
              }
              best_s[t] = cand;
              best_pl[t] = 0;
              best_pr[t] = (uint8_t)pr;
              break;
            }
          }
        }
      } else {
        uint16_t jmax = (uint16_t)((start <= L) ? start : L);
        for (uint16_t j = 1; j <= jmax; j++) {
          size_t idx_prev = span_index(start, j, L);
          npycrf_id_t prev_id = work->span_id[idx_prev];
          int16_t lm = lm_bigram_logp(&model->lm, prev_id, curr_id, curr_luni);
          npycrf_score_t add = q16_mul_q8((npycrf_score_t)model->lambda0, (npycrf_score_t)lm);
          npycrf_score_t edge = seg + add;

          size_t sid_prev = (size_t)start * L1 + (size_t)j;
          npycrf_score_t *prev_row = dp + sid_prev * (size_t)nbest;
          for (uint16_t pr = 0; pr < nbest; pr++) {
            npycrf_score_t prev = prev_row[pr];
            if (prev == NPYCRF_SCORE_NEG_INF) continue;
            npycrf_score_t cand = prev + edge;
            for (uint16_t t = 0; t < nbest; t++) {
              if (cand > best_s[t]) {
                for (uint16_t u = nbest - 1; u > t; u--) {
                  best_s[u] = best_s[u - 1];
                  best_pl[u] = best_pl[u - 1];
                  best_pr[u] = best_pr[u - 1];
                }
                best_s[t] = cand;
                best_pl[t] = (uint8_t)j;
                best_pr[t] = (uint8_t)pr;
                break;
              }
            }
          }
        }
      }

      /* store */
      size_t sid = (size_t)pos * L1 + (size_t)k;
      npycrf_score_t *dst = dp + sid * (size_t)nbest;
      uint8_t *dst_pl = bp_len + sid * (size_t)nbest;
      uint8_t *dst_pr = bp_rank + sid * (size_t)nbest;
      for (uint16_t r = 0; r < nbest; r++) {
        dst[r] = best_s[r];
        dst_pl[r] = best_pl[r];
        dst_pr[r] = best_pr[r];
      }
    }
  }

  /* 4) pick global top N from final states */
  if (nbest > 64u) return -13;
  npycrf_score_t top_s[64];
  uint16_t top_k[64];
  uint8_t top_r[64];
  for (uint16_t i = 0; i < nbest; i++) {
    top_s[i] = NPYCRF_SCORE_NEG_INF;
    top_k[i] = 0;
    top_r[i] = 0;
    out_b_count[i] = 0;
    if (out_scores) out_scores[i] = 0;
  }

  uint16_t kmax_end = (uint16_t)((n_cp <= L) ? n_cp : L);
  for (uint16_t k = 1; k <= kmax_end; k++) {
    size_t sid = (size_t)n_cp * L1 + (size_t)k;
    npycrf_score_t *row = dp + sid * (size_t)nbest;
    for (uint16_t r = 0; r < nbest; r++) {
      npycrf_score_t s = row[r];
      if (s == NPYCRF_SCORE_NEG_INF) continue;
      for (uint16_t t = 0; t < nbest; t++) {
        if (s > top_s[t]) {
          for (uint16_t u = nbest - 1; u > t; u--) {
            top_s[u] = top_s[u - 1];
            top_k[u] = top_k[u - 1];
            top_r[u] = top_r[u - 1];
          }
          top_s[t] = s;
          top_k[t] = k;
          top_r[t] = (uint8_t)r;
          break;
        }
      }
    }
  }

  /* 5) backtrack each */
  int out_n = 0;
  for (uint16_t i = 0; i < nbest; i++) {
    if (top_s[i] == NPYCRF_SCORE_NEG_INF || top_k[i] == 0) continue;

    uint16_t *bout = out_b_cp_flat + (size_t)i * out_b_cap;
    size_t bcnt = 0;
    uint16_t pos = n_cp;
    uint16_t k = top_k[i];
    uint8_t r = top_r[i];

    while (1) {
      bout[bcnt++] = pos;
      if (pos == 0) break;
      uint16_t start = (uint16_t)(pos - k);
      size_t sid = (size_t)pos * L1 + (size_t)k;
      uint8_t pl = bp_len[sid * (size_t)nbest + (size_t)r];
      uint8_t pr = bp_rank[sid * (size_t)nbest + (size_t)r];
      pos = start;
      k = (uint16_t)pl;
      r = pr;
      if (pos == 0) {
        bout[bcnt++] = 0;
        break;
      }
      if (k == 0) return -30;
      if (bcnt > (size_t)n_cp + 1u) return -31;
    }

    /* reverse */
    for (size_t a = 0; a < bcnt / 2u; a++) {
      uint16_t tmp = bout[a];
      bout[a] = bout[bcnt - 1u - a];
      bout[bcnt - 1u - a] = tmp;
    }
    if (bout[0] != 0 || bout[bcnt - 1u] != n_cp) return -32;

    out_b_count[i] = bcnt;
    if (out_scores) out_scores[i] = top_s[i];
    out_n++;
  }

  return out_n;
}
