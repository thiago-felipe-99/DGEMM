#ifndef DGEMM_AVX_UNROLL
#define DGEMM_AVX_UNROLL
void multiplyAVXUnroll(int length, double *matrixA, double *matrixB,
                       double *matrixC);
#endif
