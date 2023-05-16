#include "dgemm.h"
#include <stdlib.h>
#include <x86intrin.h>

void dgemm_simple(int length, double *a, double *b, double *c) {
  for (int i = 0; i < length; ++i) {
    for (int j = 0; j < length; ++j) {
      for (int k = 0; k < length; k++) {
        c[i * length + j] += a[i + k * length] * b[k + j * length];
      }
    }
  }
}

void dgemm_transpose(int length, double *a, double *b, double *c) {
  double *at = malloc(length * length * sizeof(double));

  for (int i = 0; i < length; ++i) {
    for (int j = 0; j < length; ++j) {
      at[i + j * length] = a[j + i * length];
    }
  }

  for (int i = 0; i < length; ++i) {
    for (int j = 0; j < length; ++j) {
      for (int k = 0; k < length; k++) {
        c[i * length + j] += at[i * length + k] * b[k + j * length];
      }
    }
  }

  free(at);
}

void dgemm_transpose_unroll(int length, double *a, double *b, double *c) {
  double *at = malloc(length * length * sizeof(double));

  for (int i = 0; i < length; i++) {
    for (int j = 0; j < length; ++j) {
      at[i + j * length] = a[j + i * length];
    }
  }

  for (int i = 0; i < length; i++) {
    int j = 0;
    for (j = 0; j < length - UNROLL; j += UNROLL) {
      for (int k = 0; k < length; k++) {
        for (int r = 0; r < UNROLL; r++) {
          c[i * length + j + r] += at[(r + i) * length + k] * b[k + j * length];
        }
      }
    }
    for (; j < length; j++) {
      for (int k = 0; k < length; k++) {
        c[i * length + j] += at[i * length + k] * b[k + j * length];
      }
    }
  }

  free(at);
}

void dgemm_avx(int length, double *a, double *b, double *c) {
#if AVX == 256
  for (int i = 0; i < length; i += AVX_QT_DOUBLE) {
    for (int j = 0; j < length; j++) {
      __m256d acc = _mm256_load_pd(c + i + j * length);
      for (int k = 0; k < length; k++) {
        __m256d row = _mm256_broadcast_sd(a + i * length + k);
        __m256d column = _mm256_load_pd(b + k * length + j);
        __m256d mul = _mm256_mul_pd(row, column);
        acc = _mm256_add_pd(acc, mul);
      }

      _mm256_store_pd(c + i + j * length, acc);
    }
  }
#elif AVX == 512
  for (int i = 0; i < length; i += AVX_QT_DOUBLE) {
    for (int j = 0; j < length; j++) {
      __m512d acc = _mm512_load_pd(c + i + j * length);
      for (int k = 0; k < length; k++) {
        __m512d row = _mm512_broadcastsd_pd(_mm_load_sd(a + i * length + k));
        __m512d column = _mm512_load_pd(b + k * length + j);
        __m512d mul = _mm512_mul_pd(row, column);
        acc = _mm512_add_pd(acc, mul);
      }

      _mm512_store_pd(c + i + j * length, acc);
    }
  }
#else
  double *at = aligned_alloc(AVX_SIZE_DOUBLE, length * length * sizeof(double));

  for (int i = 0; i < length; ++i) {
    for (int j = 0; j < length; ++j) {
      at[i + j * length] = a[j + i * length];
    }
  }

  for (int i = 0; i < length; i++) {
    for (int j = 0; j < length; j += AVX_QT_DOUBLE) {
      for (int k = 0; k < length; k++) {
        c[i * length + j + 0] += at[i * length + k] * b[k + (j + 0) * length];
        c[i * length + j + 1] += at[i * length + k] * b[k + (j + 1) * length];
        c[i * length + j + 2] += at[i * length + k] * b[k + (j + 2) * length];
        c[i * length + j + 3] += at[i * length + k] * b[k + (j + 3) * length];
      }
    }
  }

  free(at);
#endif
}

void dgemm_avx_unroll(int length, double *a, double *b, double *c) {
#if AVX == 256
  int i = 0;
  for (; i < length; i += UNROLL * AVX_QT_DOUBLE) {
    for (int j = 0; j < length; j++) {
      __m256d acc[UNROLL];

      for (int r = 0; r < UNROLL; r++)
        acc[r] = _mm256_load_pd(c + i + j * length + r * AVX_QT_DOUBLE);

      for (int k = 0; k < length; k++) {
        __m256d column = _mm256_broadcast_sd(a + k + j * length);

        for (int r = 0; r < UNROLL; r++) {
          __m256d row = _mm256_load_pd(b + k * length + i + r * AVX_QT_DOUBLE);
          __m256d mul = _mm256_mul_pd(column, row);
          acc[r] = _mm256_add_pd(acc[r], mul);
        }
      }

      for (int r = 0; r < UNROLL; r++)
        _mm256_store_pd(c + i + j * length + r * AVX_QT_DOUBLE, acc[r]);
    }
  }
  for (; i < length; i++) {
    for (int j = 0; j < length; j += AVX_QT_DOUBLE) {
      __m256d acc = _mm256_load_pd(c + i * length + j);

      for (int k = 0; k < length; k++) {
        __m256d row = _mm256_broadcast_sd(b + i * length + k);
        __m256d column = _mm256_load_pd(a + k * length + j);
        __m256d mul = _mm256_mul_pd(row, column);
        acc = _mm256_add_pd(acc, mul);
      }

      _mm256_store_pd(c + i * length + j, acc);
    }
  }
#elif AVX == 512
  int i = 0;
  for (; i < length; i += UNROLL * AVX_QT_DOUBLE) {
    for (int j = 0; j < length; j++) {
      __m512d acc[UNROLL];

      for (int r = 0; r < UNROLL; r++)
        acc[r] = _mm512_load_pd(c + i + j * length + r * AVX_QT_DOUBLE);

      for (int k = 0; k < length; k++) {
        __m512d column = _mm512_broadcastsd_pd(_mm_load_sd(a + j * length + k));

        for (int r = 0; r < UNROLL; r++) {
          __m512d row = _mm512_load_pd(b + k * length + i + r * AVX_QT_DOUBLE);
          __m512d mul = _mm512_mul_pd(column, row);
          acc[r] = _mm512_add_pd(acc[r], mul);
        }
      }

      for (int r = 0; r < UNROLL; r++)
        _mm512_store_pd(c + i + j * length + r * AVX_QT_DOUBLE, acc[r]);
    }
  }
  for (; i < length; i += AVX_QT_DOUBLE) {
    for (int j = 0; j < length; j++) {
      __m512d acc = _mm512_load_pd(c + j * length + i);
      for (int k = 0; k < length; k++) {
        __m512d column = _mm512_broadcastsd_pd(_mm_load_sd(a + j * length + k));
        __m512d row = _mm512_load_pd(b + k * length + i);
        __m512d mul = _mm512_mul_pd(column, row);
        acc = _mm512_add_pd(acc, mul);
      }

      _mm512_store_pd(c + j * length + i, acc);
    }
  }
#else
  double *at = malloc(length * length * sizeof(double));

  for (int i = 0; i < length; i++) {
    for (int j = 0; j < length; ++j) {
      at[i + j * length] = a[j + i * length];
    }
  }

  for (int i = 0; i < length; i++) {
    int j = 0;
    for (j = 0; j < length - UNROLL * AVX_QT_DOUBLE;
         j += UNROLL * AVX_QT_DOUBLE) {
      for (int k = 0; k < length; k++) {
        for (int r = 0; r < UNROLL; r++) {
          c[j + r + 0 + i * length] +=
              at[k + (0 + j + r) * length] * b[k + i * length];
          c[j + r + 1 + i * length] +=
              at[k + (1 + j + r) * length] * b[k + i * length];
          c[j + r + 2 + i * length] +=
              at[k + (2 + j + r) * length] * b[k + i * length];
          c[j + r + 3 + i * length] +=
              at[k + (3 + j + r) * length] * b[k + i * length];
          c[j + r + 4 + i * length] +=
              at[k + (4 + j + r) * length] * b[k + i * length];
          c[j + r + 5 + i * length] +=
              at[k + (5 + j + r) * length] * b[k + i * length];
          c[j + r + 6 + i * length] +=
              at[k + (6 + j + r) * length] * b[k + i * length];
          c[j + r + 7 + i * length] +=
              at[k + (7 + j + r) * length] * b[k + i * length];
        }
      }
    }
    for (; j < length; j++) {
      for (int k = 0; k < length; k++) {
        c[j + 0 + i * length] += at[k + (0 + j) * length] * b[k + i * length];
        c[j + 1 + i * length] += at[k + (1 + j) * length] * b[k + i * length];
        c[j + 2 + i * length] += at[k + (2 + j) * length] * b[k + i * length];
        c[j + 3 + i * length] += at[k + (3 + j) * length] * b[k + i * length];
        c[j + 4 + i * length] += at[k + (4 + j) * length] * b[k + i * length];
        c[j + 5 + i * length] += at[k + (5 + j) * length] * b[k + i * length];
        c[j + 6 + i * length] += at[k + (6 + j) * length] * b[k + i * length];
        c[j + 7 + i * length] += at[k + (7 + j) * length] * b[k + i * length];
      }
    }
  }

  free(at);
#endif
}

void block_avx_unroll_blocking(int length, int si, int sj, int sk, double *a,
                               double *b, double *c) {
#if AVX == 256
  for (int i = 0; i < si + BLOCK_SIZE; i += UNROLL * AVX_QT_DOUBLE) {
    for (int j = 0; j < sj + BLOCK_SIZE; j++) {
      __m256d acc[UNROLL];

      for (int r = 0; r < UNROLL; r++)
        acc[r] = _mm256_load_pd(c + i + j * length + r * AVX_QT_DOUBLE);

      for (int k = 0; k < sk + BLOCK_SIZE; k++) {
        __m256d column = _mm256_broadcast_sd(a + k + j * length);

        for (int r = 0; r < UNROLL; r++) {
          __m256d row = _mm256_load_pd(b + k * length + i + r * AVX_QT_DOUBLE);
          __m256d mul = _mm256_mul_pd(column, row);
          acc[r] = _mm256_add_pd(acc[r], mul);
        }
      }

      for (int r = 0; r < UNROLL; r++)
        _mm256_store_pd(c + i + j * length + r * AVX_QT_DOUBLE, acc[r]);
    }
  }
#elif AVX == 512
  for (int i = 0; i < si + BLOCK_SIZE; i += UNROLL * AVX_QT_DOUBLE) {
    for (int j = 0; j < sj + BLOCK_SIZE; j++) {
      __m512d acc[UNROLL];

      for (int r = 0; r < UNROLL; r++)
        acc[r] = _mm512_load_pd(c + i + j * length + r * AVX_QT_DOUBLE);

      for (int k = 0; k < sk + BLOCK_SIZE; k++) {
        __m512d column = _mm512_broadcastsd_pd(_mm_load_sd(a + j * length + k));

        for (int r = 0; r < UNROLL; r++) {
          __m512d row = _mm512_load_pd(b + k * length + i + r * AVX_QT_DOUBLE);
          __m512d mul = _mm512_mul_pd(column, row);
          acc[r] = _mm512_add_pd(acc[r], mul);
        }
      }

      for (int r = 0; r < UNROLL; r++)
        _mm512_store_pd(c + i + j * length + r * AVX_QT_DOUBLE, acc[r]);
    }
  }
#else
  double *at = malloc(length * length * sizeof(double));

  for (int i = 0; i < length; i++) {
    for (int j = 0; j < length; ++j) {
      at[i + j * length] = a[j + i * length];
    }
  }

  for (int i = 0; i < length; i++) {
    int j = 0;
    for (j = 0; j < length - UNROLL * AVX_QT_DOUBLE;
         j += UNROLL * AVX_QT_DOUBLE) {
      for (int k = 0; k < length; k++) {
        for (int r = 0; r < UNROLL; r++) {
          c[j + r + 0 + i * length] +=
              at[k + (0 + j + r) * length] * b[k + i * length];
          c[j + r + 1 + i * length] +=
              at[k + (1 + j + r) * length] * b[k + i * length];
          c[j + r + 2 + i * length] +=
              at[k + (2 + j + r) * length] * b[k + i * length];
          c[j + r + 3 + i * length] +=
              at[k + (3 + j + r) * length] * b[k + i * length];
          c[j + r + 4 + i * length] +=
              at[k + (4 + j + r) * length] * b[k + i * length];
          c[j + r + 5 + i * length] +=
              at[k + (5 + j + r) * length] * b[k + i * length];
          c[j + r + 6 + i * length] +=
              at[k + (6 + j + r) * length] * b[k + i * length];
          c[j + r + 7 + i * length] +=
              at[k + (7 + j + r) * length] * b[k + i * length];
        }
      }
    }
  }

  free(at);
#endif
}

void dgemm_avx_unroll_blocking(int length, double *a, double *b, double *c) {
  for (int sj = 0; sj < length; sj += BLOCK_SIZE)
    for (int si = 0; si < length; si += BLOCK_SIZE)
      for (int sk = 0; sk < length; sk += BLOCK_SIZE)
        block_avx_unroll_blocking(length, si, sj, sk, a, b, c);
}
