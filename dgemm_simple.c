#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define EXIT_FAILURE 1

const double range = 4096;

void multiplyMatrix(int length, double *matrixA, double *matrixB,
                    double *matrixC) {
  for (int i = 0; i < length; ++i) {
    for (int j = 0; j < length; ++j) {
      // cij = C[i][j]
      double cij = matrixC[i + j * length];

      for (int k = 0; k < length; k++) {
        // cij += A[i][k]*B[k][j]
        cij += matrixA[i + k * length] * matrixB[k + j * length];
      }

      matrixC[i + j * length] = cij;
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

  double *matrixA = malloc(length * length * sizeof(double));
  double *matrixB = malloc(length * length * sizeof(double));
  double *matrixC = malloc(length * length * sizeof(double));

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
