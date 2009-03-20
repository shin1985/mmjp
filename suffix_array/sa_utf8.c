/*
 * sa_utf8.c
 *
 * UTF-8 対応接尾辞配列の実装
 */

#include "sa_utf8.h"

/* ---------------- UTF-8 ヘルパー関数 ---------------- */

/* UTF-8 継続バイトかどうかを判定 */
static inline int sa_is_utf8_cont(uint8_t b) {
  return (b & 0xC0u) == 0x80u;
}

/* 先頭バイトから UTF-8 シーケンス長を取得 */
static inline size_t sa_utf8_seq_len_from_lead(uint8_t lead) {
  if ((lead & 0x80u) == 0x00u) return 1;           /* 0xxxxxxx: ASCII */
  if ((lead & 0xE0u) == 0xC0u) return 2;           /* 110xxxxx: 2 バイト */
  if ((lead & 0xF0u) == 0xE0u) return 3;           /* 1110xxxx: 3 バイト */
  if ((lead & 0xF8u) == 0xF0u) return 4;           /* 11110xxx: 4 バイト */
  return 1;                                        /* 無効な先頭バイト */
}

/*
 * 最小限の検証: 必要な継続バイトが存在し、10xxxxxx 形式であることを確認
 * 無効な場合は、スキャンを頑健に保つために 1 バイトとして扱う
 */
static size_t sa_utf8_advance(const uint8_t *s, size_t remaining, unsigned flags) {
  if (remaining == 0) return 0;
  const uint8_t lead = s[0];
  size_t n = sa_utf8_seq_len_from_lead(lead);
  if (n > remaining) return 1;
  if (!(flags & SA_BUILD_VALIDATE_UTF8)) return n;

  /* 継続バイトの検証 */
  if (n == 1) {
    return sa_is_utf8_cont(lead) ? 1 : 1; /* 単独の継続バイトはバイトとして扱う */
  }
  for (size_t i = 1; i < n; ++i) {
    if (!sa_is_utf8_cont(s[i])) return 1;
  }
  /*
   * オーバーロング/サロゲート/>U+10FFFF の拒否は必要に応じて追加可能だが、
   * 小さいフットプリントのために省略。接尾辞配列境界には継続バイト検証で通常十分
   */
  return n;
}

/* ASCII 空白文字かどうかを判定 */
static inline int sa_is_ascii_space(uint8_t c) {
  return (c == ' ') || (c == '\t') || (c == '\r') || (c == '\n');
}

/* ASCII 句読点かどうかを判定（範囲: 0x21-0x2F, 0x3A-0x40, 0x5B-0x60, 0x7B-0x7E） */
static inline int sa_is_ascii_punct(uint8_t c) {
  return (c >= 0x21u && c <= 0x2Fu) ||
         (c >= 0x3Au && c <= 0x40u) ||
         (c >= 0x5Bu && c <= 0x60u) ||
         (c >= 0x7Bu && c <= 0x7Eu);
}

/* ---------------- 接尾辞配列構築 ---------------- */

/* 指定フラグでの接尾辞開始位置数をカウント */
size_t sa_utf8_count_starts(const uint8_t *text, size_t text_len, unsigned flags) {
  if (!text) return 0;
  size_t n = 0;
  size_t pos = 0;
  while (pos < text_len) {
    const uint8_t lead = text[pos];
    size_t adv = sa_utf8_advance(text + pos, text_len - pos, flags);

    int skip = 0;
    if ((flags & SA_BUILD_SKIP_ASCII_SPACE) && lead < 0x80u && sa_is_ascii_space(lead)) skip = 1;
    if ((flags & SA_BUILD_SKIP_ASCII_PUNCT) && lead < 0x80u && sa_is_ascii_punct(lead)) skip = 1;

    if (!skip && !sa_is_utf8_cont(lead)) {
      ++n;
    }
    pos += (adv ? adv : 1);
  }
  return n;
}

/* (suffix_start + depth) 位置の文字を取得。終端を超えたら -1 を返す */
static inline int sa_char_at(const uint8_t *text, size_t text_len, sa_idx_t start, size_t depth) {
  size_t p = (size_t)start + depth;
  if (p >= text_len) return -1;
  return (int)text[p];
}

/* 3 つの値の中央値を返す */
static inline int sa_median3(int a, int b, int c) {
  if (a < b) {
    if (b < c) return b;
    return (a < c) ? c : a;
  } else {
    if (a < c) return a;
    return (b < c) ? c : b;
  }
}

/* 2 つの接尾辞を比較 */
static int sa_compare_suffix(const uint8_t *text, size_t text_len,
                             sa_idx_t a, sa_idx_t b, size_t depth) {
  size_t pa = (size_t)a + depth;
  size_t pb = (size_t)b + depth;
  while (pa < text_len && pb < text_len) {
    uint8_t ca = text[pa];
    uint8_t cb = text[pb];
    if (ca != cb) return (ca < cb) ? -1 : 1;
    ++pa; ++pb;
  }
  if (pa == text_len && pb == text_len) return 0;
  return (pa == text_len) ? -1 : 1;
}

/* 挿入ソート（小さい範囲用） */
static void sa_insertion_sort(sa_idx_t *sa, size_t l, size_t r, size_t depth,
                              const uint8_t *text, size_t text_len) {
  for (size_t i = l + 1; i < r; ++i) {
    sa_idx_t v = sa[i];
    size_t j = i;
    while (j > l && sa_compare_suffix(text, text_len, v, sa[j - 1], depth) < 0) {
      sa[j] = sa[j - 1];
      --j;
    }
    sa[j] = v;
  }
}

/* 挿入ソートに切り替える閾値 */
#ifndef SA_SORT_INSERTION_THRESHOLD
#define SA_SORT_INSERTION_THRESHOLD 16u
#endif

/* タスクスタックの最大サイズ */
#ifndef SA_SORT_STACK_MAX
#define SA_SORT_STACK_MAX 64u
#endif

/* ソートタスク構造体 */
typedef struct {
  size_t l;
  size_t r;
  size_t depth;
} sa_task_t;

/*
 * 反復的 3-way 基数クイックソート
 *
 * 成功時は 1 を返す。
 * 明示的タスクスタックがオーバーフローした場合は 0 を返す（呼び出し側は構築失敗として扱う）
 */
static int sa_sort_3way_radix(sa_idx_t *sa, size_t n,
                              const uint8_t *text, size_t text_len) {
  if (n < 2) return 1;

  sa_task_t stack[SA_SORT_STACK_MAX];
  size_t sp = 0;

  sa_task_t cur;
  cur.l = 0;
  cur.r = n;
  cur.depth = 0;

  for (;;) {
    while (cur.r - cur.l > SA_SORT_INSERTION_THRESHOLD) {
      const size_t l = cur.l;
      const size_t r = cur.r;
      const size_t d = cur.depth;

      const size_t mid = l + (r - l) / 2;
      const int a = sa_char_at(text, text_len, sa[l], d);
      const int b = sa_char_at(text, text_len, sa[mid], d);
      const int c = sa_char_at(text, text_len, sa[r - 1], d);
      const int v = sa_median3(a, b, c);

      size_t lt = l;
      size_t gt = r - 1;
      size_t i = l;
      while (i <= gt) {
        const int t = sa_char_at(text, text_len, sa[i], d);
        if (t < v) {
          sa_idx_t tmp = sa[lt];
          sa[lt] = sa[i];
          sa[i] = tmp;
          ++lt;
          ++i;
        } else if (t > v) {
          sa_idx_t tmp = sa[i];
          sa[i] = sa[gt];
          sa[gt] = tmp;
          if (gt == 0) break; /* 安全対策 */
          --gt;
        } else {
          ++i;
        }
      }

      /* セグメント: [l,lt) < v, [lt,gt+1) == v, (gt+1,r) > v */
      const size_t less_l = l;
      const size_t less_r = lt;
      const size_t eq_l = lt;
      const size_t eq_r = gt + 1;
      const size_t gr_l = gt + 1;
      const size_t gr_r = r;

      /*
       * 1 つのセグメントを継続（末尾再帰除去）し、他をプッシュ
       * スタックを小さく保つために最大セグメントを選択
       */
      size_t seg1_l = less_l, seg1_r = less_r, seg1_d = d;
      size_t seg2_l = gr_l,   seg2_r = gr_r,   seg2_d = d;
      size_t seg3_l = eq_l,   seg3_r = eq_r,   seg3_d = (v == -1) ? d : (d + 1);

      /*
       * v == -1 の場合、すべての接尾辞がこの深さで終了 -> 等価セグメントは既に最終
       * サイズ 0 として扱い、処理しない
       */
      if (v == -1) {
        seg3_l = seg3_r = 0;
      }

      /* サイズ計算 */
      const size_t sz1 = (seg1_r > seg1_l) ? (seg1_r - seg1_l) : 0;
      const size_t sz2 = (seg2_r > seg2_l) ? (seg2_r - seg2_l) : 0;
      const size_t sz3 = (seg3_r > seg3_l) ? (seg3_r - seg3_l) : 0;

      /* 最大を選択 */
      sa_task_t next = {0, 0, 0};
      int picked = 0;
      if (sz1 >= sz2 && sz1 >= sz3) {
        next.l = seg1_l; next.r = seg1_r; next.depth = seg1_d; picked = 1;
      } else if (sz2 >= sz1 && sz2 >= sz3) {
        next.l = seg2_l; next.r = seg2_r; next.depth = seg2_d; picked = 2;
      } else {
        next.l = seg3_l; next.r = seg3_r; next.depth = seg3_d; picked = 3;
      }

      /* サイズ > 1 の他のセグメントをプッシュ */
      if (picked != 1 && sz1 > 1) {
        if (sp >= SA_SORT_STACK_MAX) return 0;
        stack[sp++] = (sa_task_t){seg1_l, seg1_r, seg1_d};
      }
      if (picked != 2 && sz2 > 1) {
        if (sp >= SA_SORT_STACK_MAX) return 0;
        stack[sp++] = (sa_task_t){seg2_l, seg2_r, seg2_d};
      }
      if (picked != 3 && sz3 > 1) {
        if (sp >= SA_SORT_STACK_MAX) return 0;
        stack[sp++] = (sa_task_t){seg3_l, seg3_r, seg3_d};
      }

      cur = next;
      if (cur.r - cur.l <= 1) break;
    }

    if (cur.r - cur.l > 1) {
      sa_insertion_sort(sa, cur.l, cur.r, cur.depth, text, text_len);
    }

    if (sp == 0) break;
    cur = stack[--sp];
  }

  return 1;
}

/* 接尾辞配列を構築 */
size_t sa_utf8_build(sa_idx_t *sa_out, size_t sa_out_cap,
                     const uint8_t *text, size_t text_len,
                     unsigned flags) {
  if (!sa_out || sa_out_cap == 0 || !text) return 0;

  /* 接尾辞開始位置を収集 */
  size_t n = 0;
  size_t pos = 0;
  while (pos < text_len) {
    const uint8_t lead = text[pos];
    size_t adv = sa_utf8_advance(text + pos, text_len - pos, flags);

    int skip = 0;
    if ((flags & SA_BUILD_SKIP_ASCII_SPACE) && lead < 0x80u && sa_is_ascii_space(lead)) skip = 1;
    if ((flags & SA_BUILD_SKIP_ASCII_PUNCT) && lead < 0x80u && sa_is_ascii_punct(lead)) skip = 1;

    if (!skip && !sa_is_utf8_cont(lead)) {
      if (n >= sa_out_cap) return 0; /* 容量不足 */
      sa_out[n++] = (sa_idx_t)pos;
    }
    pos += (adv ? adv : 1);
  }

  if (n == 0) return 0;

  /* ソート */
  if (!sa_sort_3way_radix(sa_out, n, text, text_len)) {
    return 0;
  }

  return n;
}

/* ---------------- 接尾辞とキーの比較 ---------------- */

/*
 * 辞書順比較: suffix(at pos_off) vs key
 *   < 0: suffix < key
 *   = 0: equal
 *   > 0: suffix > key
 */
static int sa_cmp_suffix_key_lex(const sa_utf8_view_t *v, sa_idx_t s, size_t pos_off,
                                 const uint8_t *key, size_t key_len) {
  size_t p = (size_t)s + pos_off;
  size_t i = 0;
  while (i < key_len && p < v->text_len) {
    uint8_t sc = v->text[p];
    uint8_t kc = key[i];
    if (sc < kc) return -1;
    if (sc > kc) return 1;
    ++p;
    ++i;
  }

  if (i == key_len) {
    /* キー消費完了。接尾辞も消費済み -> 等価、そうでなければ suffix > key */
    return (p == v->text_len) ? 0 : 1;
  }

  /* キーより先に接尾辞が消費された */
  return -1;
}

/* プレフィックス比較: 接尾辞が key で始まるかどうか */
static int sa_cmp_suffix_key_prefix(const sa_utf8_view_t *v, sa_idx_t s, size_t pos_off,
                                    const uint8_t *key, size_t key_len) {
  size_t p = (size_t)s + pos_off;
  size_t i = 0;
  while (i < key_len && p < v->text_len) {
    uint8_t sc = v->text[p];
    uint8_t kc = key[i];
    if (sc != kc) return (sc < kc) ? -1 : 1;
    ++p;
    ++i;
  }

  if (i == key_len) {
    /* キー消費完了。接尾辞も消費済み -> 等価、そうでなければ suffix > key */
    return (p == v->text_len) ? 0 : 1;
  }

  /* 接尾辞が先に消費された */
  return -1;
}

/*
 * prefix key より厳密に大きい最小のバイト文字列を計算
 * プレフィックス範囲クエリの標準的なテクニック:
 *   upper = lower_bound(next_key)
 * next_key は末尾バイトをインクリメント（繰り上げあり）して切り詰めたもの
 *
 * tmp に書き込まれた next_key の長さを返す（<= key_len）。next_key が存在しない場合は 0
 */
static size_t sa_make_next_key(uint8_t *tmp, size_t tmp_cap, const uint8_t *key, size_t key_len) {
  if (!tmp || tmp_cap == 0 || !key || key_len == 0) return 0;
  if (key_len > tmp_cap) key_len = tmp_cap;

  /* コピー */
  for (size_t i = 0; i < key_len; ++i) tmp[i] = key[i];

  /* 末尾から繰り上げでインクリメント */
  for (size_t i = key_len; i-- > 0;) {
    if (tmp[i] != 0xFFu) {
      tmp[i] = (uint8_t)(tmp[i] + 1u);
      return i + 1; /* インクリメント後のバイトで切り詰め */
    }
  }
  return 0; /* すべて 0xFF */
}

/* 辞書順での下限を見つける */
static size_t sa_lower_bound_lex(const sa_utf8_view_t *v,
                                 const uint8_t *key, size_t key_len,
                                 size_t pos_off,
                                 size_t begin, size_t end) {
  size_t lo = begin;
  size_t hi = end;
  while (lo < hi) {
    const size_t mid = lo + (hi - lo) / 2;
    const sa_idx_t s = v->sa[mid];
    const int cmp = sa_cmp_suffix_key_lex(v, s, pos_off, key, key_len);
    if (cmp < 0) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

/* プレフィックス出現数をカウント */
size_t sa_utf8_count_prefix(const sa_utf8_view_t *view,
                            const uint8_t *key, size_t key_len) {
  if (!view || !view->text || !view->sa || !key || key_len == 0) return 0;
  if (view->sa_len == 0) return 0;

  /* key の下限 */
  const size_t low = sa_lower_bound_lex(view, key, key_len, 0, 0, view->sa_len);

  /* next_key による上限 */
  uint8_t next_key_buf[64];
  uint8_t *tmp = next_key_buf;
  size_t tmp_cap = sizeof(next_key_buf);

  /*
   * 長いキーの場合、インクリメント位置が切り詰め領域内であれば小さいバッファでも正確な境界を得られる
   * シンプルかつ安全に保つため、スタック上では 64 バイトまで。
   * それ以上長い場合は遅いパスにフォールバック
   */
  if (key_len > tmp_cap) {
    /* 遅いパス: プレフィックス比較器による二分探索で上限を見つける */
    size_t lo = low;
    size_t hi = view->sa_len;
    while (lo < hi) {
      const size_t mid = lo + (hi - lo) / 2;
      const sa_idx_t s = view->sa[mid];
      const int cmp_pref = sa_cmp_suffix_key_prefix(view, s, 0, key, key_len);
      if (cmp_pref <= 0) {
        /*
         * 接尾辞がプレフィックス key を持つか < key => 右に移動する必要あり
         * 注: cmp_pref は key が接尾辞のプレフィックスの場合に 0 を返す。
         * 辞書順ではそのような接尾辞は >= key なので、上限では右に移動しつつプレフィックスが一致
         */
        lo = mid + 1;
      } else {
        hi = mid;
      }
    }
    return (lo > low) ? (lo - low) : 0;
  }

  const size_t next_len = sa_make_next_key(tmp, tmp_cap, key, key_len);
  if (next_len == 0) {
    /* 表現可能な次のキーがない => 一致する接尾辞はすべて末尾にある */
    /* 実際に一致するものだけをカウント */
    size_t i = low;
    while (i < view->sa_len) {
      if (sa_cmp_suffix_key_prefix(view, view->sa[i], 0, key, key_len) != 0) break;
      ++i;
    }
    return (i > low) ? (i - low) : 0;
  }

  const size_t high = sa_lower_bound_lex(view, tmp, next_len, 0, low, view->sa_len);

  /*
   * [low, high) 範囲はバイト辞書順で prefix=key を保証されているが、
   * key に 0xFF が含まれる場合（UTF-8 では通常起こらない）のために念のため補正
   */
  size_t cnt = (high > low) ? (high - low) : 0;

  /* オプションの安全対策: 端を検証して（稀な）エッジケースでの偽陽性を回避 */
  if (cnt) {
    /* low から縮小 */
    size_t l = low;
    while (l < high && sa_cmp_suffix_key_prefix(view, view->sa[l], 0, key, key_len) != 0) ++l;
    /* high から縮小 */
    size_t h = high;
    while (h > l && sa_cmp_suffix_key_prefix(view, view->sa[h - 1], 0, key, key_len) != 0) --h;
    cnt = (h > l) ? (h - l) : 0;
  }

  return cnt;
}

/* 範囲内でのプレフィックス出現数をカウント（内部用） */
static size_t sa_count_prefix_in_range(const sa_utf8_view_t *v,
                                       const uint8_t *key, size_t key_len,
                                       size_t pos_off,
                                       size_t begin, size_t end) {
  if (begin >= end) return 0;
  if (key_len == 0) return 0;

  const size_t low = sa_lower_bound_lex(v, key, key_len, pos_off, begin, end);

  uint8_t next_key_buf[64];
  if (key_len > sizeof(next_key_buf)) {
    /* 範囲内での遅い上限 */
    size_t lo = low;
    size_t hi = end;
    while (lo < hi) {
      const size_t mid = lo + (hi - lo) / 2;
      const sa_idx_t s = v->sa[mid];
      const int cmp_pref = sa_cmp_suffix_key_prefix(v, s, pos_off, key, key_len);
      if (cmp_pref <= 0) lo = mid + 1;
      else hi = mid;
    }
    /* 検証 */
    size_t l = low;
    while (l < lo && sa_cmp_suffix_key_prefix(v, v->sa[l], pos_off, key, key_len) != 0) ++l;
    size_t h = lo;
    while (h > l && sa_cmp_suffix_key_prefix(v, v->sa[h - 1], pos_off, key, key_len) != 0) --h;
    return (h > l) ? (h - l) : 0;
  }

  const size_t next_len = sa_make_next_key(next_key_buf, sizeof(next_key_buf), key, key_len);
  size_t high;
  if (next_len == 0) {
    high = end;
  } else {
    high = sa_lower_bound_lex(v, next_key_buf, next_len, pos_off, low, end);
  }

  size_t l = low;
  while (l < high && sa_cmp_suffix_key_prefix(v, v->sa[l], pos_off, key, key_len) != 0) ++l;
  size_t h = high;
  while (h > l && sa_cmp_suffix_key_prefix(v, v->sa[h - 1], pos_off, key, key_len) != 0) --h;
  return (h > l) ? (h - l) : 0;
}

/* バイグラム出現数をカウント */
sa_bigram_count_t sa_utf8_count_bigram(const sa_utf8_view_t *view,
                                       const uint8_t *forward, size_t forward_len,
                                       const uint8_t *back, size_t back_len) {
  sa_bigram_count_t res;
  res.forward = 0;
  res.forward_back = 0;
  if (!view || !view->text || !view->sa) return res;
  if (!forward || forward_len == 0) return res;

  /* 前方語の範囲を見つける */
  /* 下限 */
  const size_t low = sa_lower_bound_lex(view, forward, forward_len, 0, 0, view->sa_len);

  /* 上限 */
  uint8_t next_fwd[64];
  size_t high = view->sa_len;
  if (forward_len <= sizeof(next_fwd)) {
    const size_t nf = sa_make_next_key(next_fwd, sizeof(next_fwd), forward, forward_len);
    if (nf) high = sa_lower_bound_lex(view, next_fwd, nf, 0, low, view->sa_len);
  } else {
    /* 遅い: 上限をスキャン */
    size_t lo2 = low;
    size_t hi2 = view->sa_len;
    while (lo2 < hi2) {
      size_t mid = lo2 + (hi2 - lo2) / 2;
      int cmp_pref = sa_cmp_suffix_key_prefix(view, view->sa[mid], 0, forward, forward_len);
      if (cmp_pref <= 0) lo2 = mid + 1;
      else hi2 = mid;
    }
    high = lo2;
  }

  /* 検証して正確なプレフィックス一致に補正 */
  size_t l = low;
  while (l < high && sa_cmp_suffix_key_prefix(view, view->sa[l], 0, forward, forward_len) != 0) ++l;
  size_t h = high;
  while (h > l && sa_cmp_suffix_key_prefix(view, view->sa[h - 1], 0, forward, forward_len) != 0) --h;

  res.forward = (h > l) ? (h - l) : 0;
  if (res.forward == 0 || !back || back_len == 0) return res;

  /* [l,h) 内で、オフセット forward_len での back をカウント */
  res.forward_back = sa_count_prefix_in_range(view, back, back_len, forward_len, l, h);
  return res;
}

/* ---------------- ユーティリティ: N コードポイントのプレフィックスをコピー ---------------- */

size_t sa_utf8_copy_prefix_n(const uint8_t *text, size_t text_len,
                             size_t start, size_t n_codepoints,
                             char *out, size_t out_cap,
                             unsigned flags) {
  if (!out || out_cap == 0) return 0;
  out[0] = '\0';
  if (!text || start >= text_len) return 0;

  size_t written = 0;
  size_t pos = start;
  size_t cps = 0;
  while (pos < text_len && cps < n_codepoints) {
    const size_t adv = sa_utf8_advance(text + pos, text_len - pos, flags);
    const size_t take = (adv ? adv : 1);
    if (written + take + 1 > out_cap) break;
    for (size_t i = 0; i < take; ++i) out[written + i] = (char)text[pos + i];
    written += take;
    pos += take;
    ++cps;
  }
  out[written] = '\0';
  return written;
}
