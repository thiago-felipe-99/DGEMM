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

#ifndef UNROLL
#define UNROLL 16
#endif

#ifndef AVX
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

void multiplyAVXUnroll(int length, double *matrixA, double *matrixB,
                       double *matrixC) {
#if AVX == 256
  int i = 0;
  for (; i < length; i += UNROLL * AVX_QT_DOUBLE) {
    for (int j = 0; j < length; j++) {
      __m256d acc[UNROLL];

      for (int r = 0; r < UNROLL; r++)
        acc[r] = _mm256_load_pd(matrixC + i + j * length + r * AVX_QT_DOUBLE);

      for (int k = 0; k < length; k++) {
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
  for (; i < length; i++) {
    for (int j = 0; j < length; j += AVX_QT_DOUBLE) {
      __m256d acc = _mm256_load_pd(matrixC + i * length + j);

      for (int k = 0; k < length; k++) {
        __m256d row = _mm256_broadcast_sd(matrixA + i * length + k);
        __m256d column = _mm256_load_pd(matrixB + k * length + j);
        __m256d mul = _mm256_mul_pd(row, column);
        acc = _mm256_add_pd(acc, mul);
      }

      _mm256_store_pd(matrixC + i * length + j, acc);
    }
  }
#elif AVX == 512
  int i = 0;
  for (; i < length; i += UNROLL * AVX_QT_DOUBLE) {
    for (int j = 0; j < length; j++) {
      __m512d acc[UNROLL];

      for (int r = 0; r < UNROLL; r++)
        acc[r] = _mm512_load_pd(matrixC + i + j * length + r * AVX_QT_DOUBLE);

      for (int k = 0; k < length; k++) {
        __m512d column =
            _mm512_broadcast_pd(_mm_load_sd(matrixB + j * length + k));

        for (int r = 0; r < UNROLL; r++) {
          __m512d row =
              _mm512_load_pd(matrixA + k * length + i + r * AVX_QT_DOUBLE);
          __m512d mul = _mm512_mul_pd(column, row);
          acc[r] = _mm512_add_pd(acc[r], mul);
        }
      }

      for (int r = 0; r < UNROLL; r++)
        _mm512_store_pd(matrixC + i + j * length + r * AVX_QT_DOUBLE, acc[r]);
    }
  }
  for (; i < length; i += AVX_QT_DOUBLE) {
    for (int j = 0; j < length; j++) {
      __m512d acc = _mm512_load_pd(matrixC + j * length + i);
      for (int k = 0; k < length; k++) {
        __m512d column =
            _mm512_broadcast_pd(_mm_load_sd(matrixB + j * length + k));
        __m512d row = _mm512_load_pd(matrixA + k * length + i);
        __m512d mul = _mm512_mul_pd(column, row);
        acc = _mm512_add_pd(acc, mul);
      }

      _mm512_store_pd(matrixC + j * length + i, acc);
    }
  }
#else
  double *temp = malloc(length * length * sizeof(double));

  for (int i = 0; i < length; i++) {
    for (int j = 0; j < length; ++j) {
      temp[i + j * length] = matrixB[j + i * length];
    }
  }

  for (int i = 0; i < length; i++) {
    int j = 0;
    for (j = 0; j < length - UNROLL * AVX_QT_DOUBLE;
         j += UNROLL * AVX_QT_DOUBLE) {
      for (int k = 0; k < length; k++) {
        for (int r = 0; r < UNROLL; r++) {
          matrixC[j + r + 0 + i * length] +=
              temp[k + (0 + j + r) * length] * matrixA[k + i * length];
          matrixC[j + r + 1 + i * length] +=
              temp[k + (1 + j + r) * length] * matrixA[k + i * length];
          matrixC[j + r + 2 + i * length] +=
              temp[k + (2 + j + r) * length] * matrixA[k + i * length];
          matrixC[j + r + 3 + i * length] +=
              temp[k + (3 + j + r) * length] * matrixA[k + i * length];
          matrixC[j + r + 4 + i * length] +=
              temp[k + (4 + j + r) * length] * matrixA[k + i * length];
          matrixC[j + r + 5 + i * length] +=
              temp[k + (5 + j + r) * length] * matrixA[k + i * length];
          matrixC[j + r + 6 + i * length] +=
              temp[k + (6 + j + r) * length] * matrixA[k + i * length];
          matrixC[j + r + 7 + i * length] +=
              temp[k + (7 + j + r) * length] * matrixA[k + i * length];
        }
      }
    }
    for (; j < length; j++) {
      for (int k = 0; k < length; k++) {
        matrixC[j + 0 + i * length] +=
            temp[k + (0 + j) * length] * matrixA[k + i * length];
        matrixC[j + 1 + i * length] +=
            temp[k + (1 + j) * length] * matrixA[k + i * length];
        matrixC[j + 2 + i * length] +=
            temp[k + (2 + j) * length] * matrixA[k + i * length];
        matrixC[j + 3 + i * length] +=
            temp[k + (3 + j) * length] * matrixA[k + i * length];
        matrixC[j + 4 + i * length] +=
            temp[k + (4 + j) * length] * matrixA[k + i * length];
        matrixC[j + 5 + i * length] +=
            temp[k + (5 + j) * length] * matrixA[k + i * length];
        matrixC[j + 6 + i * length] +=
            temp[k + (6 + j) * length] * matrixA[k + i * length];
        matrixC[j + 7 + i * length] +=
            temp[k + (7 + j) * length] * matrixA[k + i * length];
      }
    }
  }

  free(temp);
#endif
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

  multiplyAVXUnroll(newLength, A, B, C);

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

  printResult(length, diff);

  free(matrixA);
  free(matrixB);
  free(matrixC);

  return 0;
}
