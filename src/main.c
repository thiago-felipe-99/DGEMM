#include "dgemm.h"
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#ifndef MAX_PRINT_LINE
#define MAX_PRINT_LINE 4097
#endif

int get_matrix_length(int argc, char *argv[]) {
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

void generate_randons_matrices(int length, double *a, double *b, double *c) {
  srand(time(NULL));

  for (int index = 0; index < length * length; index++) {
    a[index] = index;
    b[index] = index;
    c[index] = 0;
  }
}

void print_matrix(int length, double *matrix) {
  for (int i = 0; i < length; i++) {
    printf("|");
    for (int j = 0; j < length; j++) {
      printf("%5.0f ", matrix[i + j * length]);
    }
    printf("|\n");
  }
}

void print_result(int length, clock_t diff) {
  double seconds = ((double)diff) / CLOCKS_PER_SEC;
  double mseconds = seconds * 1000;
  double gflops = ((2 * pow(length, 3)) / pow(10, 9));
  printf("%d,%.0f,%.2f\n", length, mseconds, gflops / seconds);
}

void copy_to_big_matrix(int old_length, int new_length, double *old_a,
                        double *new_a, double *old_b, double *new_b,
                        double *old_c, double *new_c) {

  int i = 0;
  for (; i < old_length; i++) {
    int j = 0;
    int io = i * old_length;
    int in = i * new_length;
    for (; j < old_length; j++) {
      new_a[j + in] = old_a[j + io];
      new_b[j + in] = old_b[j + io];
      new_c[j + in] = old_c[j + io];
    }
    for (; j < new_length; j++) {
      new_a[j + in] = 0;
      new_b[j + in] = 0;
      new_c[j + in] = 0;
    }
  }
  for (; i < new_length; i++) {
    int in = i * new_length;
    for (int j = 0; j < new_length; j++) {
      new_a[j + in] = 0;
      new_b[j + in] = 0;
      new_c[j + in] = 0;
    }
  }
}

void copy_to_small_matrix(int big_length, int small_length, double *big,
                          double *small) {
  for (int i = 0; i < small_length; i++) {
    int j = 0;
    int is = i * small_length;
    int ib = i * big_length;
    for (; j < small_length; j++) {
      small[j + is] = big[j + ib];
    }
  }
}

typedef enum {
  simple,
  transpose,
  transpose_unroll,
  avx,
  avx_unroll,
  avx_unroll_blocking
} dgemm;

void multiply(dgemm dgemm, int length, double *a, double *b, double *c) {
  int new_length = length;
  double *new_a, *new_b, *new_c;
  int factor;

  switch (dgemm) {
  case avx:
    factor = AVX_QT_DOUBLE;
    break;
  case avx_unroll:
    factor = AVX_QT_DOUBLE * UNROLL;
    break;
  case avx_unroll_blocking:
    factor = BLOCK_SIZE;
    break;
  default:
    factor = 1;
  }

  if (length % factor) {
    new_length += (factor - length % factor);

    new_a = aligned_alloc(AVX_SIZE_DOUBLE,
                          new_length * new_length * sizeof(double));
    new_b = aligned_alloc(AVX_SIZE_DOUBLE,
                          new_length * new_length * sizeof(double));
    new_c = aligned_alloc(AVX_SIZE_DOUBLE,
                          new_length * new_length * sizeof(double));

    copy_to_big_matrix(length, new_length, a, new_a, b, new_b, c, new_c);

  } else {
    new_a = a;
    new_b = b;
    new_c = c;
  }

  switch (dgemm) {
  case simple:
    dgemm_simple(new_length, new_a, new_b, new_c);
    break;
  case transpose:
    dgemm_transpose(new_length, new_a, new_b, new_c);
    break;
  case transpose_unroll:
    dgemm_transpose_unroll(new_length, new_a, new_b, new_c);
    break;
  case avx:
    dgemm_avx(new_length, new_a, new_b, new_c);
    break;
  case avx_unroll:
    dgemm_avx_unroll(new_length, new_a, new_b, new_c);
    break;
  case avx_unroll_blocking:
    dgemm_avx_unroll_blocking(new_length, new_a, new_b, new_c);
    break;
  }

  if (length % factor) {
    copy_to_small_matrix(new_length, length, new_c, c);

    free(new_a);
    free(new_b);
    free(new_c);
  }
}

int main(int argc, char *argv[]) {
  int length = get_matrix_length(argc, argv);

  double *a = aligned_alloc(AVX_SIZE_DOUBLE, length * length * sizeof(double));
  double *b = aligned_alloc(AVX_SIZE_DOUBLE, length * length * sizeof(double));
  double *c = aligned_alloc(AVX_SIZE_DOUBLE, length * length * sizeof(double));

  generate_randons_matrices(length, a, b, c);

  clock_t start = clock(), diff;
  multiply(avx_unroll, length, a, b, c);
  diff = clock() - start;

  if (length <= MAX_PRINT_LINE)
    print_matrix(length, c);

  print_result(length, diff);

  free(a);
  free(b);
  free(c);

  return 0;
}
