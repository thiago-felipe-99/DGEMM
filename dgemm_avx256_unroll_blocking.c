#include "debug.h"
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <x86intrin.h>

#define EXIT_FAILURE 1

#define AVX_QT_DOUBLE 4
#define AVX_SIZE_DOUBLE 32

#ifndef UNROLL
#define UNROLL 16
#endif

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 32
#endif

const double range = 4096;

void smallMatrix(int length, double *matrixA, double *matrixB,
                 double *matrixC) {
  double *temp = malloc(length * length * sizeof(double));

  for (int i = 0; i < length; ++i) {
    for (int j = 0; j < length; ++j) {
      temp[i + j * length] = matrixB[j + i * length];
    }
  }

  for (int i = 0; i < length; ++i) {
    for (int j = 0; j < length; ++j) {
      for (int k = 0; k < length; k++) {
        matrixC[j + i * length] +=
            temp[k + j * length] * matrixA[k + i * length];
      }
    }
  }

  free(temp);
}

void multiplyAVXUnrollBlocking(int length, int si, int sj, int sk,
                               double *matrixA, double *matrixB,
                               double *matrixC) {
  int i = 0;
  for (; i < si + BLOCK_SIZE; i += UNROLL * AVX_QT_DOUBLE) {
    for (int j = 0; j < sj + BLOCK_SIZE; j++) {
      __m256d acc[UNROLL];

      for (int r = 0; r < UNROLL; r++)
        acc[r] = _mm256_load_pd(matrixC + i + j * length + r * AVX_QT_DOUBLE);

      for (int k = sk; k < sk + BLOCK_SIZE; k++) {
        __m256d column = _mm256_broadcast_sd(matrixB + k + j * length);

        for (int r = 0; r < UNROLL; r++) {
          __m256d row =
              _mm256_load_pd(matrixA + k * length + i + r * AVX_QT_DOUBLE);
          __m256d mul = _mm256_mul_pd(column, row);
          acc[r] = _mm256_add_pd(acc[r], mul);
        }
      }

      for (int r = 0; r < UNROLL; r++)
        _mm256_store_pd(matrixC + i + j * length + r * AVX_QT_DOUBLE, acc[r]);
    }
  }
}

void multiplyMatrix(int length, double *matrixA, double *matrixB,
                    double *matrixC) {

  if (length < UNROLL) {
    smallMatrix(length, matrixA, matrixB, matrixC);
    return;
  }

  // criando matrizes que tenha fatores de tamanho igual a AVX_QT_DOUBLE *
  // UNROLL isso é necessário para multiplyAVXUnroll funcionar, porém se a
  // matriz tiver tamanho % (AVX_QT_DOUBLE * UNROLL) != 0 irá o consumir o
  // triplo de memória
  int newLength;
  double *A, *B, *C;
  if (length % (AVX_QT_DOUBLE * UNROLL)) {
    newLength =
        length - length % (AVX_QT_DOUBLE * UNROLL) + AVX_QT_DOUBLE * UNROLL;
    if (newLength % BLOCK_SIZE) {
      newLength = newLength - newLength % BLOCK_SIZE + BLOCK_SIZE;
    }
    A = aligned_alloc(AVX_SIZE_DOUBLE, newLength * newLength * sizeof(double));
    B = aligned_alloc(AVX_SIZE_DOUBLE, newLength * newLength * sizeof(double));
    C = aligned_alloc(AVX_SIZE_DOUBLE, newLength * newLength * sizeof(double));

    int i = 0;
    for (; i < length; i++) {
      int j = 0;
      int io = i * length;
      int in = i * newLength;
      for (; j < length; j++) {
        A[j + in] = matrixA[j + io];
        B[j + in] = matrixB[j + io];
        C[j + in] = 0;
      }
      for (; j < newLength; j++) {
        A[j + in] = 0;
        B[j + in] = 0;
        C[j + in] = 0;
      }
    }
    for (; i < newLength; i++) {
      int in = i * newLength;
      for (int j = 0; j < newLength; j++) {
        A[j + in] = 0;
        B[j + in] = 0;
        C[j + in] = 0;
      }
    }
  } else {
    newLength = length;
    A = matrixA;
    B = matrixB;
    C = matrixC;
  }

  for (int sj = 0; sj < newLength; sj += BLOCK_SIZE) {
    for (int si = 0; si < newLength; si += BLOCK_SIZE) {
      for (int sk = 0; sk < newLength; sk += BLOCK_SIZE) {
        multiplyAVXUnrollBlocking(newLength, si, sj, sk, A, B, C);
      }
    }
  }

  // garatindo que matrixC tenha a resposta de C e libernado memória de A, B,
  // C
  if (length % (AVX_QT_DOUBLE * UNROLL)) {
    int i = 0;
    for (; i < length; i++) {
      int j = 0;
      int io = i * length;
      int in = i * newLength;
      for (; j < length; j++) {
        matrixC[j + io] = C[j + in];
      }
    }

    free(A);
    free(B);
    free(C);
  }
}

int getMatrixLength(int argc, char *argv[]) {
  int length;
  char *endptr;

  if (argc != 2) {
    fprintf(stderr, "Error: Invalid number of arguments\n");
    return EXIT_FAILURE;
  }

  errno = 0;

  long int_val = strtol(argv[1], &endptr, 10);

  if (errno != 0 || *endptr != '\0') {
    fprintf(stderr, "Error: Invalid input\n");
    return EXIT_FAILURE;
  }

  if (int_val < INT_MIN || int_val > INT_MAX) {
    fprintf(stderr, "Error: Input out of range\n");
    return EXIT_FAILURE;
  }

  length = (int)int_val;

  return length;
}

void generateRandonsMatrix(int length, double *matrixA, double *matrixB,
                           double *matrixC) {
  srand(time(NULL));

  for (int index = 0; index < length * length; index++) {
    matrixA[index] = index;
    matrixB[index] = index;
    matrixC[index] = 0;
  }
}

int main(int argc, char *argv[]) {
  if (BLOCK_SIZE % (AVX_QT_DOUBLE * UNROLL)) {
    fprintf(stderr, "Error: Invalid BLOCK_SIZE\n");
    return EXIT_FAILURE;
  }

  int length = getMatrixLength(argc, argv);

  double *matrixA =
      aligned_alloc(AVX_SIZE_DOUBLE, length * length * sizeof(double));
  double *matrixB =
      aligned_alloc(AVX_SIZE_DOUBLE, length * length * sizeof(double));
  double *matrixC =
      aligned_alloc(AVX_SIZE_DOUBLE, length * length * sizeof(double));

  generateRandonsMatrix(length, matrixA, matrixB, matrixC);

  clock_t start = clock(), diff;
  multiplyMatrix(length, matrixA, matrixB, matrixC);
  diff = clock() - start;

  if (length < MAX_PRINT_LINE)
    printMatrix(length, matrixC);

  double seconds = ((double)diff) / CLOCKS_PER_SEC;
  double mseconds = seconds * 1000;
  double gflops = ((2 * pow(length, 3)) / pow(10, 9));
  printf("%d, %.0fms, %.2fGFLOPS/second\n", length, mseconds, gflops / seconds);

  free(matrixA);
  free(matrixB);
  free(matrixC);

  return 0;
}
