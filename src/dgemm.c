#include "dgemm.h"
#include <stdio.h>
#include <stdlib.h>
#include <x86intrin.h>

void copy_transpose(int length, double *matrix, double *transpose) {
  for (int i = 0; i < length; ++i)
    for (int j = 0; j < length; ++j)
      transpose[i + j * length] = matrix[j + i * length];
}

void transpose_matrix(int length, double *matrix) {
  for (int i = 0; i < length; i++) {
    for (int j = i; j < length; j++) {
      double temp = matrix[i + j * length];
      matrix[i + j * length] = matrix[j + i * length];
      matrix[j + i * length] = temp;
    }
  }
}

void dgemm_simple(int length, double *a, double *b, double *c) {
  for (int i = 0; i < length; ++i)
    for (int j = 0; j < length; ++j)
      for (int k = 0; k < length; k++)
        c[i * length + j] += a[i + k * length] * b[k + j * length];
}

void dgemm_simple_unroll(int length, double *a, double *b, double *c) {
  for (int i = 0; i < length; i++) {
    int j = 0;
    for (; j < length - UNROLL; j += UNROLL)
      for (int k = 0; k < length; k++)
        for (int r = 0; r < UNROLL; r++)
          c[i * length + j + r] += a[i + k * length] * b[k + (j + r) * length];

    for (; j < length; j++)
      for (int k = 0; k < length; k++)
        c[i * length + j] += a[i + k * length] * b[k + j * length];
  }
}

void block_simple_unroll(int length, int si, int sj, int sk, double *a,
                         double *b, double *c) {
  for (int i = si; i < si + BLOCK_SIZE; i++)
    for (int j = sj; j < sj + BLOCK_SIZE; j += UNROLL)
      for (int k = sk; k < sk + BLOCK_SIZE; k++)
        for (int r = 0; r < UNROLL; r++)
          c[i * length + j + r] += a[i + k * length] * b[k + (j + r) * length];
}

void dgemm_simple_unroll_blocking(int length, double *a, double *b, double *c) {
  for (int sj = 0; sj < length; sj += BLOCK_SIZE)
    for (int si = 0; si < length; si += BLOCK_SIZE)
      for (int sk = 0; sk < length; sk += BLOCK_SIZE)
        block_simple_unroll(length, si, sj, sk, a, b, c);
}

void dgemm_transpose(int length, double *a, double *b, double *c) {
  double *at = aligned_alloc(ALIGN, length * length * sizeof(double));
  copy_transpose(length, a, at);

  for (int i = 0; i < length; ++i)
    for (int j = 0; j < length; ++j)
      for (int k = 0; k < length; k++)
        c[i * length + j] += at[i * length + k] * b[k + j * length];

  free(at);
}

void dgemm_transpose_unroll(int length, double *a, double *b, double *c) {
  double *at = aligned_alloc(ALIGN, length * length * sizeof(double));
  copy_transpose(length, a, at);

  for (int i = 0; i < length; i++) {
    int j = 0;
    for (; j < length - UNROLL; j += UNROLL)
      for (int k = 0; k < length; k++)
        for (int r = 0; r < UNROLL; r++)
          c[i * length + j + r] += at[i * length + k] * b[k + (j + r) * length];

    for (; j < length; j++)
      for (int k = 0; k < length; k++)
        c[i * length + j] += at[i * length + k] * b[k + j * length];
  }

  free(at);
}

void block_transpose_unroll(int length, int si, int sj, int sk, double *at,
                            double *b, double *c) {
  for (int i = si; i < si + BLOCK_SIZE; i++)
    for (int j = sj; j < sj + BLOCK_SIZE; j += UNROLL)
      for (int k = sk; k < sk + BLOCK_SIZE; k++)
        for (int r = 0; r < UNROLL; r++)
          c[i * length + j + r] += at[i * length + k] * b[k + (j + r) * length];
}

void dgemm_transpose_unroll_blocking(int length, double *a, double *b,
                                     double *c) {
  double *at = aligned_alloc(ALIGN, length * length * sizeof(double));
  copy_transpose(length, a, at);

  for (int sj = 0; sj < length; sj += BLOCK_SIZE)
    for (int si = 0; si < length; si += BLOCK_SIZE)
      for (int sk = 0; sk < length; sk += BLOCK_SIZE)
        block_transpose_unroll(length, si, sj, sk, at, b, c);

  free(at);
}

void dgemm_simd_manual(int length, double *a, double *b, double *c) {
  double *at = aligned_alloc(ALIGN, length * length * sizeof(double));
  copy_transpose(length, a, at);

  int i = 0;
  for (; i < length - SIMD_MANUAL_QT_DOUBLE; i += SIMD_MANUAL_QT_DOUBLE) {
    for (int j = 0; j < length; j++)
      for (int k = 0; k < length; k++) {
        c[(i + 0) * length + j] += at[(i + 0) * length + k] * b[k + j * length];
        c[(i + 1) * length + j] += at[(i + 1) * length + k] * b[k + j * length];
        c[(i + 2) * length + j] += at[(i + 2) * length + k] * b[k + j * length];
        c[(i + 3) * length + j] += at[(i + 3) * length + k] * b[k + j * length];
      }
  }

  for (; i < length; ++i)
    for (int j = 0; j < length; ++j)
      for (int k = 0; k < length; k++)
        c[i * length + j] += at[i * length + k] * b[k + j * length];

  free(at);
}

void dgemm_simd_manual_unroll(int length, double *a, double *b, double *c) {
  double *at = aligned_alloc(ALIGN, length * length * sizeof(double));
  copy_transpose(length, a, at);

  for (int i = 0; i < length; i++) {
    int j = 0;
    for (; j < length; j += UNROLL * SIMD_MANUAL_QT_DOUBLE)
      for (int k = 0; k < length; k++)
        for (int r = 0; r < UNROLL; r++) {
          c[i * length + j + r + 0] +=
              at[k + i * length] * b[k + (j + r + 0) * length];
          c[i * length + j + r + 1] +=
              at[k + i * length] * b[k + (j + r + 1) * length];
          c[i * length + j + r + 2] +=
              at[k + i * length] * b[k + (j + r + 2) * length];
          c[i * length + j + r + 3] +=
              at[k + i * length] * b[k + (j + r + 3) * length];
        }
  }

  free(at);
}

void block_simd_manual_unroll(int length, int si, int sj, int sk, double *at,
                              double *b, double *c) {
  for (int i = si; i < si + BLOCK_SIZE; i++)
    for (int j = sj; j < sj + BLOCK_SIZE; j += UNROLL * SIMD_MANUAL_QT_DOUBLE)
      for (int k = sk; k < sk + BLOCK_SIZE; k++)
        for (int r = 0; r < UNROLL; r++) {
          c[i * length + j + r + 0] +=
              at[k + i * length] * b[k + (0 + j + r) * length];
          c[i * length + j + r + 1] +=
              at[k + i * length] * b[k + (1 + j + r) * length];
          c[i * length + j + r + 2] +=
              at[k + i * length] * b[k + (2 + j + r) * length];
          c[i * length + j + r + 3] +=
              at[k + i * length] * b[k + (3 + j + r) * length];
        }
}

void dgemm_simd_manual_unroll_blocking(int length, double *a, double *b,
                                       double *c) {
  double *at = aligned_alloc(ALIGN, length * length * sizeof(double));
  copy_transpose(length, a, at);

  for (int sj = 0; sj < length; sj += BLOCK_SIZE)
    for (int si = 0; si < length; si += BLOCK_SIZE)
      for (int sk = 0; sk < length; sk += BLOCK_SIZE)
        block_simd_manual_unroll(length, si, sj, sk, at, b, c);

  free(at);
}

void dgemm_avx256(int length, double *a, double *b, double *c) {
#if __AVX__ || __AVX2__
  for (int i = 0; i < length; i += AVX256_QT_DOUBLE) {
    for (int j = 0; j < length; j++) {
      __m256d acc = _mm256_load_pd(c + i + j * length);
      for (int k = 0; k < length; k++) {
        __m256d row = _mm256_load_pd(a + i + k * length);
        __m256d column = _mm256_broadcast_sd(b + k + j * length);
        __m256d mul = _mm256_mul_pd(row, column);
        acc = _mm256_add_pd(acc, mul);
      }

      _mm256_store_pd(c + i + j * length, acc);
    }
  }

  transpose_matrix(length, c);
#endif
}

void dgemm_avx256_unroll(int length, double *a, double *b, double *c) {
#if __AVX__ || __AVX2__
  for (int i = 0; i < length; i += UNROLL * AVX256_QT_DOUBLE) {
    for (int j = 0; j < length; j++) {
      __m256d acc[UNROLL];

      for (int r = 0; r < UNROLL; r++)
        acc[r] = _mm256_load_pd(c + i + j * length + r * AVX256_QT_DOUBLE);

      for (int k = 0; k < length; k++) {
        __m256d column = _mm256_broadcast_sd(b + k + j * length);

        for (int r = 0; r < UNROLL; r++) {
          __m256d row =
              _mm256_load_pd(a + i + k * length + r * AVX256_QT_DOUBLE);
          __m256d mul = _mm256_mul_pd(row, column);
          acc[r] = _mm256_add_pd(acc[r], mul);
        }
      }

      for (int r = 0; r < UNROLL; r++)
        _mm256_store_pd(c + i + j * length + r * AVX256_QT_DOUBLE, acc[r]);
    }
  }

  transpose_matrix(length, c);
#endif
}

void block_avx256_unroll(int length, int si, int sj, int sk, double *a,
                         double *b, double *c) {
#if __AVX__ || __AVX2__
  for (int i = si; i < si + BLOCK_SIZE; i += UNROLL * AVX256_QT_DOUBLE) {
    for (int j = sj; j < sj + BLOCK_SIZE; j++) {
      __m256d acc[UNROLL];

      for (int r = 0; r < UNROLL; r++)
        acc[r] = _mm256_load_pd(c + i + j * length + r * AVX256_QT_DOUBLE);

      for (int k = sk; k < sk + BLOCK_SIZE; k++) {
        __m256d column = _mm256_broadcast_sd(b + k + j * length);

        for (int r = 0; r < UNROLL; r++) {
          __m256d row =
              _mm256_load_pd(a + i + k * length + r * AVX256_QT_DOUBLE);
          __m256d mul = _mm256_mul_pd(row, column);
          acc[r] = _mm256_add_pd(acc[r], mul);
        }
      }

      for (int r = 0; r < UNROLL; r++)
        _mm256_store_pd(c + i + j * length + r * AVX256_QT_DOUBLE, acc[r]);
    }
  }
#endif
}

void dgemm_avx256_unroll_blocking(int length, double *a, double *b, double *c) {
  for (int sj = 0; sj < length; sj += BLOCK_SIZE)
    for (int si = 0; si < length; si += BLOCK_SIZE)
      for (int sk = 0; sk < length; sk += BLOCK_SIZE)
        block_avx256_unroll(length, si, sj, sk, a, b, c);

  transpose_matrix(length, c);
}

void dgemm_avx512(int length, double *a, double *b, double *c) {
#if __AVX512F__
  for (int i = 0; i < length; i += AVX512_QT_DOUBLE) {
    for (int j = 0; j < length; j++) {
      __m512d acc = _mm512_load_pd(c + i + j * length);
      for (int k = 0; k < length; k++) {
        __m512d row = _mm512_load_pd(a + i + length * k);
        __m512d column = _mm512_broadcastsd_pd(_mm_load_sd(b + k + j * length));
        __m512d mul = _mm512_mul_pd(row, column);
        acc = _mm512_add_pd(acc, mul);
      }

      _mm512_store_pd(c + i + j * length, acc);
    }
  }

  transpose_matrix(length, c);
#endif
}

void dgemm_avx512_unroll(int length, double *a, double *b, double *c) {
#if __AVX512F__
  for (int i = 0; i < length; i += UNROLL * AVX512_QT_DOUBLE) {
    for (int j = 0; j < length; j++) {
      __m512d acc[UNROLL];

      for (int r = 0; r < UNROLL; r++)
        acc[r] = _mm512_load_pd(c + i + j * length + r * AVX512_QT_DOUBLE);

      for (int k = 0; k < length; k++) {
        __m512d column = _mm512_broadcastsd_pd(_mm_load_sd(b + k + j * length));

        for (int r = 0; r < UNROLL; r++) {
          __m512d row =
              _mm512_load_pd(a + i + k * length + r * AVX512_QT_DOUBLE);
          __m512d mul = _mm512_mul_pd(row, column);
          acc[r] = _mm512_add_pd(acc[r], mul);
        }
      }

      for (int r = 0; r < UNROLL; r++)
        _mm512_store_pd(c + i + j * length + r * AVX512_QT_DOUBLE, acc[r]);
    }
  }

  transpose_matrix(length, c);
#endif
}

void block_avx512_unroll(int length, int si, int sj, int sk, double *a,
                         double *b, double *c) {
#if __AVX512F__
  for (int i = si; i < si + BLOCK_SIZE; i += UNROLL * AVX512_QT_DOUBLE) {
    for (int j = sj; j < sj + BLOCK_SIZE; j++) {
      __m512d acc[UNROLL];

      for (int r = 0; r < UNROLL; r++)
        acc[r] = _mm512_load_pd(c + i + j * length + r * AVX512_QT_DOUBLE);

      for (int k = sk; k < sk + BLOCK_SIZE; k++) {
        __m512d column = _mm512_broadcastsd_pd(_mm_load_sd(b + k + j * length));

        for (int r = 0; r < UNROLL; r++) {
          __m512d row =
              _mm512_load_pd(a + i + k * length + r * AVX512_QT_DOUBLE);
          __m512d mul = _mm512_mul_pd(row, column);
          acc[r] = _mm512_add_pd(acc[r], mul);
        }
      }

      for (int r = 0; r < UNROLL; r++)
        _mm512_store_pd(c + i + j * length + r * AVX512_QT_DOUBLE, acc[r]);
    }
  }
#endif
}

void dgemm_avx512_unroll_blocking(int length, double *a, double *b, double *c) {
  for (int sj = 0; sj < length; sj += BLOCK_SIZE)
    for (int si = 0; si < length; si += BLOCK_SIZE)
      for (int sk = 0; sk < length; sk += BLOCK_SIZE)
        block_avx512_unroll(length, si, sj, sk, a, b, c);

  transpose_matrix(length, c);
}
