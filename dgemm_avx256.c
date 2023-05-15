#include "debug.h"
#include <errno.h>
#include <immintrin.h>
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

const double range = 4096;

void smallMatrix(int length, double *matrixA, double *matrixB,
                 double *matrixC) {
  for (int i = 0; i < length; ++i) {
    for (int j = 0; j < length; ++j) {
      for (int k = 0; k < length; k++) {
        matrixC[j + i * length] +=
            matrixB[j + k * length] * matrixA[k + i * length];
      }
    }
  }
}

void multiplyAVX(int newLength, double *matrixA, double *matrixB,
                    double *matrixC) {
  for (int i = 0; i < newLength; i++) {
    for (int j = 0; j < newLength; j += AVX_QT_DOUBLE) {
      __m256d acc = _mm256_load_pd(matrixC + i * newLength + j);

      for (int k = 0; k < newLength; k++) {
        __m256d row = _mm256_broadcast_sd(matrixA + i * newLength + k);
        __m256d column = _mm256_load_pd(matrixB + k * newLength + j);
        __m256d mul = _mm256_mul_pd(row, column);
        acc = _mm256_add_pd(acc, mul);
      }

      _mm256_store_pd(matrixC + i * newLength + j, acc);
    }
  }
}

void multiplyMatrix(int length, double *matrixA, double *matrixB,
                    double *matrixC) {
  if (length < AVX_QT_DOUBLE) {
    smallMatrix(length, matrixA, matrixB, matrixC);
    return;
  }

  //criando matrizes que tenha tamanhos fatores de AVX_QT_DOUBLE
  //isso é necessário para multiplyAVX funcionar, porém se a matriz 
  //tiver tamanho % AVX_QT_DOUBLE != 0 irá o consumir o triplo de memória
  int newLength;
  double *A, *B, *C;
  if (length % AVX_QT_DOUBLE) {
    newLength = length - length % AVX_QT_DOUBLE + AVX_QT_DOUBLE;
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

  multiplyAVX(newLength, A, B, C);

  // garatindo que matrixC tenha a resposta de C e libernado memória de A, B, C
  if (length % AVX_QT_DOUBLE) {
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

  if (length <= MAX_PRINT_LINE)
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
