#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "../tools/mmjp_model.h"
#include "../npycrf_lite/npycrf_lite.h"

/* ------------------------------
 * Helpers
 * ------------------------------ */

static uint32_t xs32(uint32_t *s) {
  uint32_t x = *s;
  if (x == 0u) x = 2463534242u;
  x ^= x << 13;
  x ^= x >> 17;
  x ^= x << 5;
  *s = x;
  return x;
}

static uint32_t default_seed(void) {
  uint32_t s = (uint32_t)time(NULL);
  if (s == 0u) s = 1u;
  /* decorrelate a bit */
  (void)xs32(&s);
  return s;
}

/* ------------------------------
 * Python object
 * ------------------------------ */

typedef struct {
  PyObject_HEAD
  mmjp_loaded_model_t model;
  int model_loaded;

  uint16_t max_n_cp;

  /* work */
  npycrf_work_t wk;
  uint8_t *workbuf;
  size_t workcap;

  /* boundaries */
  uint16_t *b_cp;
  uint16_t *b_bytes;
  size_t b_cap; /* in uint16 entries */

  /* sampling buffer */
  uint8_t *samplebuf;
  size_t samplecap;

  /* nbest buffer + outputs */
  uint8_t *nbestbuf;
  size_t nbestcap;
  uint16_t *b_cp_flat;
  size_t bcp_flat_cap; /* in uint16 entries */
  size_t *bcount_arr;
  size_t bcount_cap;
  npycrf_score_t *score_arr;
  size_t score_cap;
  uint16_t nbest_last;
} PyMMJPModel;

static void PyMMJPModel_dealloc(PyMMJPModel *self) {
  if (self->model_loaded) {
    mmjp_model_free(&self->model);
    self->model_loaded = 0;
  }
  free(self->workbuf);
  free(self->b_cp);
  free(self->b_bytes);
  free(self->samplebuf);
  free(self->nbestbuf);
  free(self->b_cp_flat);
  free(self->bcount_arr);
  free(self->score_arr);
  Py_TYPE(self)->tp_free((PyObject *)self);
}

static int ensure_work(PyMMJPModel *self, Py_ssize_t utf8_len) {
  if (!self->model_loaded) {
    PyErr_SetString(PyExc_RuntimeError, "model not loaded");
    return -1;
  }

  /* Worst case: 1 byte == 1 codepoint. This is a safe upper bound. */
  uint32_t need_cp32 = (utf8_len > 0) ? (uint32_t)utf8_len : 1u;
  if (need_cp32 > 60000u) {
    /* keep within uint16_t and avoid absurd allocations */
    need_cp32 = 60000u;
  }

  uint16_t need_cp = (uint16_t)need_cp32;
  if (need_cp < 64u) need_cp = 64u;

  if (self->max_n_cp >= need_cp && self->workbuf && self->b_cp && self->b_bytes) {
    return 0;
  }

  uint16_t new_max = self->max_n_cp ? self->max_n_cp : 1024u;
  while (new_max < need_cp && new_max < 60000u) {
    uint32_t doubled = (uint32_t)new_max * 2u;
    new_max = (doubled > 60000u) ? 60000u : (uint16_t)doubled;
  }
  if (new_max < need_cp) new_max = need_cp;

  size_t need_work = npycrf_workbuf_size(new_max, self->model.m.max_word_len);
  uint8_t *new_workbuf = (uint8_t *)realloc(self->workbuf, need_work);
  if (!new_workbuf) {
    PyErr_NoMemory();
    return -1;
  }
  self->workbuf = new_workbuf;
  self->workcap = need_work;

  /* boundaries arrays (max_n_cp + 1) */
  size_t new_bcap = (size_t)new_max + 1u;
  uint16_t *new_bcp = (uint16_t *)realloc(self->b_cp, new_bcap * sizeof(uint16_t));
  uint16_t *new_bbytes = (uint16_t *)realloc(self->b_bytes, new_bcap * sizeof(uint16_t));
  if (!new_bcp || !new_bbytes) {
    free(new_bcp);
    free(new_bbytes);
    self->b_cp = NULL;
    self->b_bytes = NULL;
    PyErr_NoMemory();
    return -1;
  }
  self->b_cp = new_bcp;
  self->b_bytes = new_bbytes;
  self->b_cap = new_bcap;

  /* init work */
  int rc = npycrf_work_init(&self->wk,
                           self->workbuf,
                           self->workcap,
                           (uint16_t)new_max,
                           self->model.m.max_word_len);
  if (rc != 0) {
    PyErr_Format(PyExc_RuntimeError, "npycrf_work_init failed rc=%d", rc);
    return -1;
  }

  self->max_n_cp = new_max;

  /* invalidate derived buffers (realloc lazily) */
  self->samplecap = 0;
  self->nbestcap = 0;
  self->bcp_flat_cap = 0;
  self->bcount_cap = 0;
  self->score_cap = 0;
  self->nbest_last = 0;

  return 0;
}

static int ensure_samplebuf(PyMMJPModel *self) {
  size_t need = npycrf_samplebuf_size(self->max_n_cp, self->model.m.max_word_len);
  if (self->samplebuf && self->samplecap >= need) return 0;
  uint8_t *p = (uint8_t *)realloc(self->samplebuf, need);
  if (!p) {
    PyErr_NoMemory();
    return -1;
  }
  self->samplebuf = p;
  self->samplecap = need;
  return 0;
}

static int ensure_nbest(PyMMJPModel *self, uint16_t nbest) {
  if (nbest == 0) nbest = 1;
  size_t need = npycrf_nbestbuf_size(self->max_n_cp, self->model.m.max_word_len, nbest);
  if (!self->nbestbuf || self->nbestcap < need) {
    uint8_t *p = (uint8_t *)realloc(self->nbestbuf, need);
    if (!p) {
      PyErr_NoMemory();
      return -1;
    }
    self->nbestbuf = p;
    self->nbestcap = need;
  }

  size_t out_b_cap = (size_t)self->max_n_cp + 1u;
  size_t flat_need = out_b_cap * (size_t)nbest;
  if (!self->b_cp_flat || self->bcp_flat_cap < flat_need) {
    uint16_t *p = (uint16_t *)realloc(self->b_cp_flat, flat_need * sizeof(uint16_t));
    if (!p) {
      PyErr_NoMemory();
      return -1;
    }
    self->b_cp_flat = p;
    self->bcp_flat_cap = flat_need;
  }
  if (!self->bcount_arr || self->bcount_cap < (size_t)nbest) {
    size_t *p = (size_t *)realloc(self->bcount_arr, (size_t)nbest * sizeof(size_t));
    if (!p) {
      PyErr_NoMemory();
      return -1;
    }
    self->bcount_arr = p;
    self->bcount_cap = (size_t)nbest;
  }
  if (!self->score_arr || self->score_cap < (size_t)nbest) {
    npycrf_score_t *p = (npycrf_score_t *)realloc(self->score_arr, (size_t)nbest * sizeof(npycrf_score_t));
    if (!p) {
      PyErr_NoMemory();
      return -1;
    }
    self->score_arr = p;
    self->score_cap = (size_t)nbest;
  }

  self->nbest_last = nbest;
  return 0;
}

static PyObject *tokens_from_bbytes(const uint8_t *utf8, Py_ssize_t len,
                                   const uint16_t *b_bytes, size_t b_count) {
  if (b_count < 2) {
    return PyList_New(0);
  }
  Py_ssize_t n_tok = (Py_ssize_t)(b_count - 1);
  PyObject *out = PyList_New(n_tok);
  if (!out) return NULL;

  for (Py_ssize_t i = 0; i < n_tok; i++) {
    uint16_t s = b_bytes[i];
    uint16_t e = b_bytes[i + 1];
    if (e < s || e > (uint16_t)len) {
      Py_DECREF(out);
      PyErr_SetString(PyExc_RuntimeError, "boundary out of range");
      return NULL;
    }
    PyObject *tok = PyUnicode_DecodeUTF8((const char *)(utf8 + s), (Py_ssize_t)(e - s), "strict");
    if (!tok) {
      Py_DECREF(out);
      return NULL;
    }
    PyList_SET_ITEM(out, i, tok);
  }
  return out;
}

static PyObject *tokens_with_offsets_from_bounds(const uint8_t *utf8, Py_ssize_t len,
                                                const uint16_t *b_cp,
                                                const uint16_t *b_bytes,
                                                size_t b_count,
                                                int unit_char) {
  if (b_count < 2) {
    return PyList_New(0);
  }
  Py_ssize_t n_tok = (Py_ssize_t)(b_count - 1);
  PyObject *out = PyList_New(n_tok);
  if (!out) return NULL;

  for (Py_ssize_t i = 0; i < n_tok; i++) {
    uint16_t s_b = b_bytes[i];
    uint16_t e_b = b_bytes[i + 1];
    if (e_b < s_b || e_b > (uint16_t)len) {
      Py_DECREF(out);
      PyErr_SetString(PyExc_RuntimeError, "boundary out of range");
      return NULL;
    }

    PyObject *tok = PyUnicode_DecodeUTF8((const char *)(utf8 + s_b), (Py_ssize_t)(e_b - s_b), "strict");
    if (!tok) {
      Py_DECREF(out);
      return NULL;
    }

    long long s_off = (long long)(unit_char ? b_cp[i] : s_b);
    long long e_off = (long long)(unit_char ? b_cp[i + 1] : e_b);

    PyObject *tup = PyTuple_New(3);
    if (!tup) {
      Py_DECREF(tok);
      Py_DECREF(out);
      return NULL;
    }
    PyTuple_SET_ITEM(tup, 0, tok);
    PyTuple_SET_ITEM(tup, 1, PyLong_FromLongLong(s_off));
    PyTuple_SET_ITEM(tup, 2, PyLong_FromLongLong(e_off));
    PyList_SET_ITEM(out, i, tup);
  }
  return out;
}


/* ------------------------------
 * Methods
 * ------------------------------ */

static int PyMMJPModel_init(PyMMJPModel *self, PyObject *args, PyObject *kwargs) {
  const char *path = NULL;
  unsigned int max_n_cp = 1024u;

  static char *kwlist[] = {"model_path", "max_n_cp", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "s|I", kwlist, &path, &max_n_cp)) {
    return -1;
  }

  memset(&self->model, 0, sizeof(self->model));
  self->model_loaded = 0;

  int rc = mmjp_model_load_bin(path, &self->model);
  if (rc != 0) {
    PyErr_Format(PyExc_RuntimeError, "mmjp_model_load_bin failed rc=%d", rc);
    return -1;
  }
  self->model_loaded = 1;

  self->max_n_cp = (uint16_t)max_n_cp;
  self->workbuf = NULL;
  self->workcap = 0;
  self->b_cp = NULL;
  self->b_bytes = NULL;
  self->b_cap = 0;
  self->samplebuf = NULL;
  self->samplecap = 0;
  self->nbestbuf = NULL;
  self->nbestcap = 0;
  self->b_cp_flat = NULL;
  self->bcp_flat_cap = 0;
  self->bcount_arr = NULL;
  self->bcount_cap = 0;
  self->score_arr = NULL;
  self->score_cap = 0;
  self->nbest_last = 0;

  /* allocate initial buffers */
  if (ensure_work(self, (Py_ssize_t)max_n_cp) != 0) {
    return -1;
  }

  return 0;
}

static PyObject *PyMMJPModel_repr(PyMMJPModel *self) {
  if (!self->model_loaded) {
    return PyUnicode_FromString("<mmjp.Model (unloaded)>");
  }
  return PyUnicode_FromFormat("<mmjp.Model vocab=%u max_word_len=%u>",
                              (unsigned)self->model.m.lm.vocab_size,
                              (unsigned)self->model.m.max_word_len);
}

static PyObject *PyMMJPModel_tokenize(PyMMJPModel *self, PyObject *args, PyObject *kwargs) {
  PyObject *text_obj = NULL;
  static char *kwlist[] = {"text", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", kwlist, &text_obj)) {
    return NULL;
  }

  const uint8_t *utf8 = NULL;
  Py_ssize_t len = 0;
  if (PyUnicode_Check(text_obj)) {
    utf8 = (const uint8_t *)PyUnicode_AsUTF8AndSize(text_obj, &len);
    if (!utf8) return NULL;
  } else if (PyBytes_Check(text_obj)) {
    utf8 = (const uint8_t *)PyBytes_AS_STRING(text_obj);
    len = PyBytes_GET_SIZE(text_obj);
  } else {
    PyErr_SetString(PyExc_TypeError, "text must be str or bytes");
    return NULL;
  }

  if (ensure_work(self, len) != 0) return NULL;

  size_t b_count = 0;
  npycrf_score_t score = 0;
  int rc = npycrf_decode(&self->model.m, utf8, (size_t)len,
                         &self->wk,
                         self->b_cp, self->b_cap,
                         &b_count,
                         &score);
  if (rc != 0) {
    PyErr_Format(PyExc_RuntimeError, "npycrf_decode failed rc=%d", rc);
    return NULL;
  }

  npycrf_boundaries_cp_to_bytes(self->wk.cp_off, self->b_cp, b_count, self->b_bytes);

  return tokens_from_bbytes(utf8, len, self->b_bytes, b_count);
}

static PyObject *PyMMJPModel_tokenize_with_offsets(PyMMJPModel *self, PyObject *args, PyObject *kwargs) {
  PyObject *text_obj = NULL;
  const char *unit = "char";
  static char *kwlist[] = {"text", "unit", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|s", kwlist, &text_obj, &unit)) {
    return NULL;
  }

  int unit_char = 1;
  if (unit && (strcmp(unit, "byte") == 0 || strcmp(unit, "bytes") == 0)) {
    unit_char = 0;
  } else if (unit && (strcmp(unit, "char") == 0 || strcmp(unit, "cp") == 0 || strcmp(unit, "codepoint") == 0)) {
    unit_char = 1;
  } else if (unit && unit[0] != 0) {
    PyErr_SetString(PyExc_ValueError, "unit must be one of: 'char'/'cp'/'codepoint' or 'byte'/'bytes'");
    return NULL;
  }

  const uint8_t *utf8 = NULL;
  Py_ssize_t len = 0;
  if (PyUnicode_Check(text_obj)) {
    utf8 = (const uint8_t *)PyUnicode_AsUTF8AndSize(text_obj, &len);
    if (!utf8) return NULL;
  } else if (PyBytes_Check(text_obj)) {
    utf8 = (const uint8_t *)PyBytes_AS_STRING(text_obj);
    len = PyBytes_GET_SIZE(text_obj);
  } else {
    PyErr_SetString(PyExc_TypeError, "text must be str or bytes");
    return NULL;
  }

  if (ensure_work(self, len) != 0) return NULL;

  size_t b_count = 0;
  npycrf_score_t score = 0;
  int rc = npycrf_decode(&self->model.m, utf8, (size_t)len,
                         &self->wk,
                         self->b_cp, self->b_cap,
                         &b_count,
                         &score);
  if (rc != 0) {
    PyErr_Format(PyExc_RuntimeError, "npycrf_decode failed rc=%d", rc);
    return NULL;
  }

  npycrf_boundaries_cp_to_bytes(self->wk.cp_off, self->b_cp, b_count, self->b_bytes);

  return tokens_with_offsets_from_bounds(utf8, len, self->b_cp, self->b_bytes, b_count, unit_char);
}


static PyObject *PyMMJPModel_sample(PyMMJPModel *self, PyObject *args, PyObject *kwargs) {
  PyObject *text_obj = NULL;
  double temperature = 1.0;
  PyObject *seed_obj = Py_None;
  static char *kwlist[] = {"text", "temperature", "seed", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|dO", kwlist, &text_obj, &temperature, &seed_obj)) {
    return NULL;
  }

  const uint8_t *utf8 = NULL;
  Py_ssize_t len = 0;
  if (PyUnicode_Check(text_obj)) {
    utf8 = (const uint8_t *)PyUnicode_AsUTF8AndSize(text_obj, &len);
    if (!utf8) return NULL;
  } else if (PyBytes_Check(text_obj)) {
    utf8 = (const uint8_t *)PyBytes_AS_STRING(text_obj);
    len = PyBytes_GET_SIZE(text_obj);
  } else {
    PyErr_SetString(PyExc_TypeError, "text must be str or bytes");
    return NULL;
  }

  if (ensure_work(self, len) != 0) return NULL;
  if (ensure_samplebuf(self) != 0) return NULL;

  uint32_t seed = 0u;
  if (seed_obj == Py_None) {
    seed = default_seed();
  } else {
    unsigned long long s = PyLong_AsUnsignedLongLong(seed_obj);
    if (PyErr_Occurred()) return NULL;
    seed = (uint32_t)s;
    if (seed == 0u) seed = 1u;
  }

  size_t b_count = 0;
  npycrf_score_t score = 0;
  int rc = npycrf_decode_sample(&self->model.m, utf8, (size_t)len,
                                &self->wk,
                                self->samplebuf, self->samplecap,
                                temperature,
                                seed,
                                self->b_cp, self->b_cap,
                                &b_count,
                                &score);
  if (rc != 0) {
    PyErr_Format(PyExc_RuntimeError, "npycrf_decode_sample failed rc=%d", rc);
    return NULL;
  }
  npycrf_boundaries_cp_to_bytes(self->wk.cp_off, self->b_cp, b_count, self->b_bytes);
  return tokens_from_bbytes(utf8, len, self->b_bytes, b_count);
}

static PyObject *PyMMJPModel_nbest(PyMMJPModel *self, PyObject *args, PyObject *kwargs) {
  PyObject *text_obj = NULL;
  unsigned int nbest = 8u;
  static char *kwlist[] = {"text", "nbest", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|I", kwlist, &text_obj, &nbest)) {
    return NULL;
  }
  if (nbest == 0u) nbest = 1u;
  if (nbest > 64u) {
    PyErr_SetString(PyExc_ValueError, "nbest too large (max 64)");
    return NULL;
  }

  const uint8_t *utf8 = NULL;
  Py_ssize_t len = 0;
  if (PyUnicode_Check(text_obj)) {
    utf8 = (const uint8_t *)PyUnicode_AsUTF8AndSize(text_obj, &len);
    if (!utf8) return NULL;
  } else if (PyBytes_Check(text_obj)) {
    utf8 = (const uint8_t *)PyBytes_AS_STRING(text_obj);
    len = PyBytes_GET_SIZE(text_obj);
  } else {
    PyErr_SetString(PyExc_TypeError, "text must be str or bytes");
    return NULL;
  }

  if (ensure_work(self, len) != 0) return NULL;
  if (ensure_nbest(self, (uint16_t)nbest) != 0) return NULL;

  int rc = npycrf_decode_nbest(&self->model.m,
                              utf8, (size_t)len,
                              &self->wk,
                              self->nbestbuf, self->nbestcap,
                              (uint16_t)nbest,
                              self->b_cp_flat, (size_t)self->max_n_cp + 1u,
                              self->bcount_arr,
                              self->score_arr);
  if (rc < 0) {
    PyErr_Format(PyExc_RuntimeError, "npycrf_decode_nbest failed rc=%d", rc);
    return NULL;
  }
  size_t out_count = (size_t)rc;

  PyObject *outer = PyList_New((Py_ssize_t)out_count);
  if (!outer) return NULL;

  for (size_t i = 0; i < out_count; i++) {
    const uint16_t *bcp = self->b_cp_flat + i * ((size_t)self->max_n_cp + 1u);
    size_t bcnt = self->bcount_arr[i];
    npycrf_boundaries_cp_to_bytes(self->wk.cp_off, bcp, bcnt, self->b_bytes);
    PyObject *tokens = tokens_from_bbytes(utf8, len, self->b_bytes, bcnt);
    if (!tokens) {
      Py_DECREF(outer);
      return NULL;
    }
    PyList_SET_ITEM(outer, (Py_ssize_t)i, tokens);
  }

  return outer;
}

static PyMethodDef PyMMJPModel_methods[] = {
  {"tokenize", (PyCFunction)PyMMJPModel_tokenize, METH_VARARGS | METH_KEYWORDS,
   "tokenize(text) -> list[str]"},
  {"tokenize_with_offsets", (PyCFunction)PyMMJPModel_tokenize_with_offsets, METH_VARARGS | METH_KEYWORDS,
   "tokenize_with_offsets(text, unit='char') -> list[tuple[str,int,int]]"},
  {"sample", (PyCFunction)PyMMJPModel_sample, METH_VARARGS | METH_KEYWORDS,
   "sample(text, temperature=1.0, seed=None) -> list[str]"},
  {"nbest", (PyCFunction)PyMMJPModel_nbest, METH_VARARGS | METH_KEYWORDS,
   "nbest(text, nbest=8) -> list[list[str]]"},
  {NULL, NULL, 0, NULL},
};

static PyTypeObject PyMMJPModelType = {
  PyVarObject_HEAD_INIT(NULL, 0)
  .tp_name = "mmjp.Model",
  .tp_basicsize = sizeof(PyMMJPModel),
  .tp_itemsize = 0,
  .tp_dealloc = (destructor)PyMMJPModel_dealloc,
  .tp_flags = Py_TPFLAGS_DEFAULT,
  .tp_doc = "MMJP model",
  .tp_methods = PyMMJPModel_methods,
  .tp_init = (initproc)PyMMJPModel_init,
  .tp_new = PyType_GenericNew,
  .tp_repr = (reprfunc)PyMMJPModel_repr,
};

/* ------------------------------
 * Module
 * ------------------------------ */

static PyMethodDef module_methods[] = {
  {NULL, NULL, 0, NULL},
};

static struct PyModuleDef moduledef = {
  PyModuleDef_HEAD_INIT,
  .m_name = "_mmjp",
  .m_doc = "C extension for mmjp",
  .m_size = -1,
  .m_methods = module_methods,
};

PyMODINIT_FUNC PyInit__mmjp(void) {
  if (PyType_Ready(&PyMMJPModelType) < 0) {
    return NULL;
  }
  PyObject *m = PyModule_Create(&moduledef);
  if (!m) return NULL;

  Py_INCREF(&PyMMJPModelType);
  if (PyModule_AddObject(m, "Model", (PyObject *)&PyMMJPModelType) < 0) {
    Py_DECREF(&PyMMJPModelType);
    Py_DECREF(m);
    return NULL;
  }

  return m;
}
