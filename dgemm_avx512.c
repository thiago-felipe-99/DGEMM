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

#define AVX_QT_DOUBLE 8
#define AVX_SIZE_DOUBLE 64

const double range = 4096;

void multiplyMatrix(int length, double *matrixA, double *matrixB,
                    double *matrixC) {
  for (int i = 0; i < length; i += AVX_QT_DOUBLE) {
    for (int j = 0; j < length; j++) {
      // cij = C[i][j]
      __m512d c0 = _mm512_load_pd(matrixC + i + j * length);

      for (int k = 0; k < length; k++) {
        // cij += A[i][k]*B[k][j]
        c0 = _mm512_add_pd(
            c0, _mm512_mul_pd(_mm512_load_pd(matrixA + i + k * length),
                              _mm512_broadcastsd_pd(
                                  _mm_load_sd(matrixB + k + j * length))));
      }

      _mm512_store_pd(matrixC + i + j * length, c0);
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
    matrixA[index] = ((double)rand() / RAND_MAX) * range;
    matrixB[index] = ((double)rand() / RAND_MAX) * range;
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

  double seconds = ((double)diff) / CLOCKS_PER_SEC;
  double mseconds = seconds * 1000;
  double gflops = ((2 * pow(length, 3)) / pow(10, 9));
  printf("%d, %.0fms, %.2fGFLOPS/second\n", length, mseconds, gflops / seconds);

  free(matrixA);
  free(matrixB);
  free(matrixC);

  return 0;
}
