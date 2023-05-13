#include <errno.h>
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

void multiplyMatrix(int lenght, double *matrixA, double *matrixB,
                    double *matrixC) {
  for (int i = 0; i < lenght; i += AVX_QT_DOUBLE) {
    for (int j = 0; j < lenght; j++) {
      // cij = C[i][j]
      __m256d c0 = _mm256_load_pd(matrixC + i + j * lenght);

      for (int k = 0; k < lenght; k++) {
        // cij += A[i][k]*B[k][j]
        c0 = _mm256_add_pd(
            c0, _mm256_mul_pd(_mm256_load_pd(matrixA + i + k * lenght),
                              _mm256_broadcast_sd(matrixB + k + j * lenght)));
      }

      _mm256_store_pd(matrixC + i + j * lenght, c0);
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

  int mseconds = diff * 1000 / CLOCKS_PER_SEC;
  printf("Time to calculate matrix %dx%d: %dms\n", length, length, mseconds);

  free(matrixA);
  free(matrixB);
  free(matrixC);

  return 0;
}
