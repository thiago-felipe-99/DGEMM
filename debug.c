#include <stdio.h>

void printMatrix(int length, double *matrix) {
  for (int i = 0; i < length; i++) {
    printf("|");
    for (int j = 0; j < length; j++) {
      printf("%5.0f ", matrix[i + j * length]);
    }
    printf("|\n");
  }
}

