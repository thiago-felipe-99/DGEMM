#include "dgemm.h"
#include <errno.h>
#include <float.h>
#include <getopt.h>
#include <limits.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef MAX_PRINT_LINE
#define MAX_PRINT_LINE 0
#endif

typedef enum {
  simple,
  transpose,
  transpose_unroll,
  avx,
  avx_unroll,
  avx_unroll_blocking,
  DGEMM_COUNT
} dgemm;

void process_dgemms(char *option, bool dgemms[]) {
  char *token;
  const char delimiter[] = ",";
  token = strtok(option, delimiter);
  while (token != NULL) {

    if (strcmp(token, "simple") == 0)
      dgemms[simple] = true;
    else if (strcmp(token, "transpose") == 0)
      dgemms[transpose] = true;
    else if (strcmp(token, "transpose_unroll") == 0)
      dgemms[transpose_unroll] = true;
    else if (strcmp(token, "avx") == 0)
      dgemms[avx] = true;
    else if (strcmp(token, "avx_unroll") == 0)
      dgemms[avx_unroll] = true;
    else if (strcmp(token, "avx_unroll_blocking") == 0)
      dgemms[avx_unroll_blocking] = true;

    token = strtok(NULL, delimiter);
  }
}

int process_length(char *option) {
  char *endptr;
  errno = 0;

  long int_val = strtol(option, &endptr, 10);

  if (errno != 0 || *endptr != '\0') {
    fprintf(stderr, "Error: Invalid input\n");
    exit(EXIT_FAILURE);
  }

  if (int_val < INT_MIN || int_val > INT_MAX) {
    fprintf(stderr, "Error: Input out of range\n");
    exit(EXIT_FAILURE);
  }

  return (int)int_val;
}

void print_help() {}

void parse_options(int argc, char *argv[], bool dgemms[], int *length,
                   bool *random) {
  struct option long_options[] = {{"dgemm", required_argument, NULL, 'd'},
                                  {"length", required_argument, NULL, 'l'},
                                  {"random", no_argument, NULL, 'r'},
                                  {"help", no_argument, NULL, 'h'},
                                  {NULL, 0, NULL, 0}};

  bool is_set_dgemms = false, is_set_length = false;

  int option;
  while ((option = getopt_long(argc, argv, "d:l:rh", long_options, NULL)) !=
         -1) {
    switch (option) {
    case 'd':
      process_dgemms(optarg, dgemms);
      is_set_dgemms = true;
      break;
    case 'l':
      *length = process_length(optarg);
      is_set_length = true;
      break;
    case 'r':
      *random = true;
      break;
    case 'h':
      print_help();
      exit(0);
    default:
      fprintf(stderr, "Error: Invalid option\n");
      exit(EXIT_FAILURE);
    }
  }

  if (!is_set_length || !is_set_dgemms) {
    print_help();
    exit(EXIT_FAILURE);
  }
}

void generate_matrices(int length, double *a, double *b, bool random) {
  srand(time(NULL));

  if (random) {
    for (int index = 0; index < length * length; index++) {
      a[index] = (double)4 * rand() / RAND_MAX;
      b[index] = (double)4 * rand() / RAND_MAX;
    }
  } else {
    for (int index = 0; index < length * length; index++) {
      a[index] = index;
      b[index] = index;
    }
  }
}

void clean_matrix(int length, double *a) {
  for (int index = 0; index < length * length; index++) {
    a[index] = 0;
  }
}

void print_matrix(int length, double *matrix) {
  for (int i = 0; i < length; i++) {
    printf("|");
    for (int j = 0; j < length; j++) {
      printf("%5.2f ", matrix[i + j * length]);
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
  case DGEMM_COUNT:
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
  bool dgemms[DGEMM_COUNT];
  int length;
  bool random;

  parse_options(argc, argv, dgemms, &length, &random);

  double *a = aligned_alloc(AVX_SIZE_DOUBLE, length * length * sizeof(double));
  double *b = aligned_alloc(AVX_SIZE_DOUBLE, length * length * sizeof(double));
  double *c = aligned_alloc(AVX_SIZE_DOUBLE, length * length * sizeof(double));

  generate_matrices(length, a, b, random);

  for (int i = 0; i < DGEMM_COUNT; i++) {
    if (dgemms[i]) {
      clean_matrix(length, c);

      clock_t start = clock(), diff;
      multiply(i, length, a, b, c);
      diff = clock() - start;

      if (length <= MAX_PRINT_LINE)
        print_matrix(length, c);

      print_result(length, diff);
    }
  }

  free(a);
  free(b);
  free(c);

  return 0;
}
