/*
 * mmjp_export_c.c
 *
 * model.bin (MMJP) を MCU で使いやすい C ヘッダに変換する。
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mmjp_model.h"

static void usage(const char *prog) {
  fprintf(stderr,
          "Usage: %s --model model.bin --out model.h [--symbol mmjp]\n"
          "  --symbol S  ... base symbol name prefix (default: mmjp)\n",
          prog);
}

static void emit_array_u32(FILE *o, const char *name, const uint32_t *a, size_t n) {
  fprintf(o, "static const uint32_t %s[%zu] = {\n", name, n);
  for (size_t i = 0; i < n; i++) {
    fprintf(o, "  %uu,%s", (unsigned)a[i], (i % 8 == 7) ? "\n" : " ");
  }
  fprintf(o, "\n};\n\n");
}

static void emit_array_i16(FILE *o, const char *name, const int16_t *a, size_t n) {
  fprintf(o, "static const int16_t %s[%zu] = {\n", name, n);
  for (size_t i = 0; i < n; i++) {
    fprintf(o, "  %d,%s", (int)a[i], (i % 12 == 11) ? "\n" : " ");
  }
  fprintf(o, "\n};\n\n");
}

static void emit_array_da_index(FILE *o, const char *name, const da_index_t *a, size_t n) {
  fprintf(o, "static const da_index_t %s[%zu] = {\n", name, n);
  for (size_t i = 0; i < n; i++) {
    fprintf(o, "  %d,%s", (int)a[i], (i % 12 == 11) ? "\n" : " ");
  }
  fprintf(o, "\n};\n\n");
}

int main(int argc, char **argv) {
  const char *model_path = NULL;
  const char *out_path = NULL;
  const char *sym = "mmjp";

  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
      model_path = argv[++i];
    } else if (strcmp(argv[i], "--out") == 0 && i + 1 < argc) {
      out_path = argv[++i];
    } else if (strcmp(argv[i], "--symbol") == 0 && i + 1 < argc) {
      sym = argv[++i];
    } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
      usage(argv[0]);
      return 0;
    } else {
      fprintf(stderr, "unknown arg: %s\n", argv[i]);
      usage(argv[0]);
      return 1;
    }
  }

  if (!model_path || !out_path) {
    usage(argv[0]);
    return 1;
  }

  mmjp_loaded_model_t lm;
  int rc = mmjp_model_load_bin(model_path, &lm);
  if (rc != 0) {
    fprintf(stderr, "load failed: %d\n", rc);
    return 1;
  }

  FILE *o = fopen(out_path, "wb");
  if (!o) {
    fprintf(stderr, "failed to open out\n");
    mmjp_model_free(&lm);
    return 1;
  }

  fprintf(o, "#pragma once\n\n");
  fprintf(o, "/* Auto-generated from %s */\n\n", model_path);
  fprintf(o, "#include <stdint.h>\n");
  fprintf(o, "#include \"npycrf_lite.h\"\n\n");

  char name_base[128], name_check[128], name_uni[128], name_fkey[128], name_fw[128], name_model[128];
  snprintf(name_base, sizeof(name_base), "%s_base", sym);
  snprintf(name_check, sizeof(name_check), "%s_check", sym);
  snprintf(name_uni, sizeof(name_uni), "%s_logp_uni", sym);
  snprintf(name_fkey, sizeof(name_fkey), "%s_feat_key", sym);
  snprintf(name_fw, sizeof(name_fw), "%s_feat_w", sym);
  snprintf(name_model, sizeof(name_model), "%s_model", sym);

  emit_array_da_index(o, name_base, lm.m.lm.trie.base, lm.m.lm.trie.capacity);
  emit_array_da_index(o, name_check, lm.m.lm.trie.check, lm.m.lm.trie.capacity);
  emit_array_i16(o, name_uni, lm.m.lm.logp_uni, lm.m.lm.vocab_size);
  if (lm.m.crf.feat_count > 0) {
    emit_array_u32(o, name_fkey, lm.m.crf.feat_key, lm.m.crf.feat_count);
    emit_array_i16(o, name_fw, lm.m.crf.feat_w, lm.m.crf.feat_count);
  } else {
    fprintf(o, "static const uint32_t %s[1] = {0};\n", name_fkey);
    fprintf(o, "static const int16_t %s[1] = {0};\n\n", name_fw);
  }

  fprintf(o,
          "static const npycrf_model_t %s = {\n"
          "  .lm = {\n"
          "    .trie = { .base = %s, .check = %s, .capacity = %zu },\n"
          "    .logp_uni = %s,\n"
          "    .logp_bi = %s,\n"
          "    .bigram_key = %s,\n"
          "    .bigram_size = %uu,\n"
          "    .vocab_size = %uu,\n"
          "    .unk_base = %d,\n"
          "    .unk_per_cp = %d,\n"
          "  },\n"
          "  .lambda0 = %d,\n"
          "  .crf = {\n"
          "    .trans00 = %d, .trans01 = %d, .trans10 = %d, .trans11 = %d,\n"
          "    .bos_to1 = %d,\n"
          "    .feat_key = %s,\n"
          "    .feat_w = %s,\n"
          "    .feat_count = %uu,\n"
          "  },\n"
          "  .max_word_len = %uu,\n"
          "};\n\n",
          name_model,
          name_base,
          name_check,
          (size_t)lm.m.lm.trie.capacity,
          name_uni,
          (lm.m.lm.logp_bi ? "(const int16_t*)0" : "(const int16_t*)0"),
          (lm.m.lm.bigram_key ? "(const uint32_t*)0" : "(const uint32_t*)0"),
          (unsigned)lm.m.lm.bigram_size,
          (unsigned)lm.m.lm.vocab_size,
          (int)lm.m.lm.unk_base,
          (int)lm.m.lm.unk_per_cp,
          (int)lm.m.lambda0,
          (int)lm.m.crf.trans00,
          (int)lm.m.crf.trans01,
          (int)lm.m.crf.trans10,
          (int)lm.m.crf.trans11,
          (int)lm.m.crf.bos_to1,
          name_fkey,
          name_fw,
          (unsigned)lm.m.crf.feat_count,
          (unsigned)lm.m.max_word_len);

  fclose(o);
  mmjp_model_free(&lm);

  fprintf(stderr, "wrote %s\n", out_path);
  return 0;
}
