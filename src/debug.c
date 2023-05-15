#include <math.h>
#include <stdio.h>
#include <time.h>

void printMatrix(int length, double *matrix) {
  for (int i = 0; i < length; i++) {
    printf("|");
    for (int j = 0; j < length; j++) {
      printf("%5.0f ", matrix[i + j * length]);
    }
    printf("|\n");
  }
}

void printResult(int length, clock_t diff) {
  double seconds = ((double)diff) / CLOCKS_PER_SEC;
  double mseconds = seconds * 1000;
  double gflops = ((2 * pow(length, 3)) / pow(10, 9));
  printf("%d,%.0f,%.2f\n", length, mseconds, gflops / seconds);
}
