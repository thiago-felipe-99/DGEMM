#include <errno.h>
#include <immintrin.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <x86intrin.h>

#define EXIT_FAILURE 1

#define AVX_QT_DOUBLE 4
#define AVX_SIZE_DOUBLE 32

const double range = 4096;

void printMatrix(int length, double *matrix) {
  for (int i = 0; i < length; i++) {
    printf("|");
    for (int j = 0; j < length; j++) {
      printf("%5.0f ", matrix[i + j * length]);
    }
    printf("|\n");
  }
}

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

void multiplyMatrix(int length, double *matrixA, double *matrixB,
                    double *matrixC) {
  if (length < AVX_QT_DOUBLE) {
    smallMatrix(length, matrixA, matrixB, matrixC);
    return;
  }

  int i = 0;
  for (i = 0; i < length; i++) {
    int j = 0;
    for (j = 0; j < length; j += AVX_QT_DOUBLE) {
      __m256d acc = _mm256_load_pd(matrixC + i * length + j);

      for (int k = 0; k < length; k++) {
        __m256d row = _mm256_broadcast_sd(matrixA + i * length + k);
        __m256d column = _mm256_load_pd(matrixB + k * length + j);
        __m256d mul = _mm256_mul_pd(row, column);
        acc = _mm256_add_pd(acc, mul);
      }

      _mm256_store_pd(matrixC + i * length + j, acc);
    }
    for (; j < length; j++) {
      for (int k = 0; k < length; k++) {
        matrixC[j + i * length] +=
            matrixB[j + k * length] * matrixA[k + i * length];
      }
    }
  }
  for (; i < length; ++i) {
    for (int j = 0; j < length; ++j) {
      for (int k = 0; k < length; k++) {
        matrixC[j + i * length] +=
            matrixB[j + k * length] * matrixA[k + i * length];
      }
    }
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

  if (length <= 127)
    printMatrix(length, matrixC);

  int mseconds = diff * 1000 / CLOCKS_PER_SEC;
  printf("Time to calculate matrix %dx%d: %dms\n", length, length, mseconds);

  free(matrixA);
  free(matrixB);
  free(matrixC);

  return 0;
}
