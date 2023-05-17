#ifndef DGEMM_H
#define DGEMM_H

#ifndef UNROLL
#define UNROLL 16
#endif

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 32
#endif

#define AVX256_QT_DOUBLE 4
#define AVX512_QT_DOUBLE 8
#define SIMD_MANUAL_QT_DOUBLE 4
#define ALIGN 64

void dgemm_simple(int length, double *a, double *b, double *c);
void dgemm_transpose(int length, double *a, double *b, double *c);
void dgemm_simd_manual(int length, double *a, double *b, double *c);
void dgemm_avx256(int length, double *a, double *b, double *c);
void dgemm_avx512(int length, double *a, double *b, double *c);
void dgemm_simple_unroll(int length, double *a, double *b, double *c);
void dgemm_transpose_unroll(int length, double *a, double *b, double *c);
void dgemm_simd_manual_unroll(int length, double *a, double *b, double *c);
void dgemm_avx256_unroll(int length, double *a, double *b, double *c);
void dgemm_avx512_unroll(int length, double *a, double *b, double *c);
void dgemm_simple_unroll_blocking(int length, double *a, double *b, double *c);
void dgemm_transpose_unroll_blocking(int length, double *a, double *b, double *c);
void dgemm_simd_manual_unroll_blocking(int length, double *a, double *b, double *c);
void dgemm_avx256_unroll_blocking(int length, double *a, double *b, double *c);
void dgemm_avx512_unroll_blocking(int length, double *a, double *b, double *c);

#endif
