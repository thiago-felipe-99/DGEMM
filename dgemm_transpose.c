#include "debug.h"
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define EXIT_FAILURE 1

const double range = 4096;

void multiplyMatrix(int length, double *matrixA, double *matrixB,
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

  double *matrixA = malloc(length * length * sizeof(double));
  double *matrixB = malloc(length * length * sizeof(double));
  double *matrixC = malloc(length * length * sizeof(double));

  generateRandonsMatrix(length, matrixA, matrixB, matrixC);

  clock_t start = clock(), diff;
  multiplyMatrix(length, matrixA, matrixB, matrixC);
  diff = clock() - start;

  if (length <= MAX_PRINT_LINE)
    printMatrix(length, matrixC);

  printResult(length, diff);

  free(matrixA);
  free(matrixB);
  free(matrixC);

  return 0;
}
