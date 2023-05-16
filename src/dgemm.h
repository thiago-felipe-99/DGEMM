#ifndef DGEMM_H
#define DGEMM_H

#ifndef UNROLL
#define UNROLL 16
#endif

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 32
#endif

#ifdef __AVX512F__
#define AVX 512
#define AVX_QT_DOUBLE 8
#define AVX_SIZE_DOUBLE 64
#elif __AVX2__ || __AVX__
#define AVX 256
#define AVX_QT_DOUBLE 4
#define AVX_SIZE_DOUBLE 32
#else
#define AVX 0
#define AVX_QT_DOUBLE 8
#define AVX_SIZE_DOUBLE 32
#endif

void dgemm_simple(int length, double *a, double *b, double *c);
void dgemm_transpose(int length, double *a, double *b, double *c);
void dgemm_transpose_unroll(int length, double *a, double *b, double *c);
void dgemm_avx(int length, double *a, double *b, double *c);
void dgemm_avx_unroll(int length, double *a, double *b, double *c);
void dgemm_avx_unroll_blocking(int length, double *a, double *b, double *c);

#endif
