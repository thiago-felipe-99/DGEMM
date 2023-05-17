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

typedef enum {
  simple,
  transpose,
  transpose_unroll,
  simd_manual,
  avx256,
  avx512,
  simd_manual_unroll,
  avx256_unroll,
  avx512_unroll,
  simd_manual_unroll_blocking,
  avx256_unroll_blocking,
  avx512_unroll_blocking,
  DGEMM_COUNT
} dgemm;

const char *dgemm_names[DGEMM_COUNT] = {
    "simple",
    "transpose",
    "transpose_unroll",
    "simd_manual",
    "avx256",
    "avx512",
    "simd_manual_unroll",
    "avx256_unroll",
    "avx512_unroll",
    "simd_manual_unroll_blocking",
    "avx256_unroll_blocking",
    "avx512_unroll_blocking",
};

void process_dgemms(char *option, bool dgemms[]) {
  char *token;
  const char delimiter[] = ",";
  token = strtok(option, delimiter);
  while (token != NULL) {

    if (strcmp(token, dgemm_names[simple]) == 0)
      dgemms[simple] = true;
    else if (strcmp(token, dgemm_names[transpose]) == 0)
      dgemms[transpose] = true;
    else if (strcmp(token, dgemm_names[transpose_unroll]) == 0)
      dgemms[transpose_unroll] = true;
    else if (strcmp(token, dgemm_names[simd_manual]) == 0)
      dgemms[simd_manual] = true;
    else if (strcmp(token, dgemm_names[avx256]) == 0)
      dgemms[avx256] = true;
    else if (strcmp(token, dgemm_names[avx512]) == 0)
      dgemms[avx512] = true;
    else if (strcmp(token, dgemm_names[simd_manual_unroll]) == 0)
      dgemms[simd_manual_unroll] = true;
    else if (strcmp(token, dgemm_names[avx256_unroll]) == 0)
      dgemms[avx256_unroll] = true;
    else if (strcmp(token, dgemm_names[avx512_unroll]) == 0)
      dgemms[avx512_unroll] = true;
    else if (strcmp(token, dgemm_names[simd_manual_unroll_blocking]) == 0)
      dgemms[simd_manual_unroll_blocking] = true;
    else if (strcmp(token, dgemm_names[avx256_unroll_blocking]) == 0)
      dgemms[avx256_unroll_blocking] = true;
    else if (strcmp(token, dgemm_names[avx512_unroll_blocking]) == 0)
      dgemms[avx512_unroll_blocking] = true;
    else {
      fprintf(stderr, "Error: Invalid dgemm '%s'\n", token);
      exit(EXIT_FAILURE);
    }

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
                   bool *random, bool *show_result) {
  struct option long_options[] = {{"dgemm", required_argument, NULL, 'd'},
                                  {"length", required_argument, NULL, 'l'},
                                  {"random", no_argument, NULL, 'r'},
                                  {"show-result", no_argument, NULL, 's'},
                                  {"help", no_argument, NULL, 'h'},
                                  {NULL, 0, NULL, 0}};

  bool is_set_dgemms = false, is_set_length = false;

  int option;
  while ((option = getopt_long(argc, argv, "d:l:rhs", long_options, NULL)) !=
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
    case 's':
      *show_result = true;
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

void print_result(dgemm dgemm, int length, clock_t diff) {
  double seconds = ((double)diff) / CLOCKS_PER_SEC;
  double mseconds = seconds * 1000;
  double gflops = ((2 * pow(length, 3)) / pow(10, 9));
  printf("%s,%d,%.0f,%.2f\n", dgemm_names[dgemm], length, mseconds,
         gflops / seconds);
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
  case simd_manual:
    factor = SIMD_MANUAL_QT_DOUBLE;
    break;
  case avx256:
    factor = AVX256_QT_DOUBLE;
    break;
  case avx512:
    factor = AVX512_QT_DOUBLE;
    break;
  case simd_manual_unroll:
    factor = SIMD_MANUAL_QT_DOUBLE * UNROLL;
    break;
  case avx256_unroll:
    factor = AVX256_QT_DOUBLE * UNROLL;
    break;
  case avx512_unroll:
    factor = AVX512_QT_DOUBLE * UNROLL;
    break;
  case simd_manual_unroll_blocking:
  case avx256_unroll_blocking:
  case avx512_unroll_blocking:
    factor = BLOCK_SIZE;
    break;
  default:
    factor = 1;
  }

  if (length % factor) {
    new_length += (factor - length % factor);

    new_a = aligned_alloc(ALIGN, new_length * new_length * sizeof(double));
    new_b = aligned_alloc(ALIGN, new_length * new_length * sizeof(double));
    new_c = aligned_alloc(ALIGN, new_length * new_length * sizeof(double));

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
  case simd_manual:
    dgemm_simd_manual(new_length, new_a, new_b, new_c);
    break;
  case avx256:
    dgemm_avx256(new_length, new_a, new_b, new_c);
    break;
  case avx512:
    dgemm_avx512(new_length, new_a, new_b, new_c);
    break;
  case simd_manual_unroll:
    dgemm_simd_manual_unroll(new_length, new_a, new_b, new_c);
    break;
  case avx256_unroll:
    dgemm_avx256_unroll(new_length, new_a, new_b, new_c);
    break;
  case avx512_unroll:
    dgemm_avx512_unroll(new_length, new_a, new_b, new_c);
    break;
  case simd_manual_unroll_blocking:
    dgemm_simd_manual_unroll_blocking(new_length, new_a, new_b, new_c);
    break;
  case avx256_unroll_blocking:
    dgemm_avx256_unroll_blocking(new_length, new_a, new_b, new_c);
    break;
  case avx512_unroll_blocking:
    dgemm_avx512_unroll_blocking(new_length, new_a, new_b, new_c);
    break;
  }

  if (length % factor) {
    copy_to_small_matrix(new_length, length, new_c, c);

    free(new_a);
    free(new_b);
    free(new_c);
  }
}

int checkAVXOrAVX2Support() {
  int cpuInfo[4];

  __asm__ volatile("cpuid"
                   : "=a"(cpuInfo[0]), "=b"(cpuInfo[1]), "=c"(cpuInfo[2]),
                     "=d"(cpuInfo[3])
                   : "a"(1));

  return ((cpuInfo[2] & (1 << 28)) || (cpuInfo[1] & (1 << 5)));
}

int checkAVX512Support() {
  int cpuInfo[4];

  __asm__ volatile("cpuid"
                   : "=a"(cpuInfo[0]), "=b"(cpuInfo[1]), "=c"(cpuInfo[2]),
                     "=d"(cpuInfo[3])
                   : "a"(7), "c"(0));

  return (cpuInfo[1] & (1 << 16));
}

void check_avx(bool dgemms[DGEMM_COUNT]) {
#if __AVX__ || __AVX2__
  if (!checkAVXOrAVX2Support()) {
    fprintf(stderr, "Error: Binary use AVX256\n");
    exit(EXIT_FAILURE);
  }
#else
  if (dgemms[avx256] || dgemms[avx256_unroll] ||
      dgemms[avx256_unroll_blocking]) {
    fprintf(stderr, "Error: Binary does not include AVX256\n");
    exit(EXIT_FAILURE);
  }
#endif

#if __AVX512F__
  if (!checkAVX512Support()) {
    fprintf(stderr, "Error: Binary use AVX512\n");
    exit(EXIT_FAILURE);
  }
#else
  if (dgemms[avx512] || dgemms[avx512_unroll] ||
      dgemms[avx512_unroll_blocking]) {
    fprintf(stderr, "Error: Binary does not include AVX512\n");
    exit(EXIT_FAILURE);
  }
#endif
}

int main(int argc, char *argv[]) {
  bool dgemms[DGEMM_COUNT];
  int length = 0;
  bool random = false, show_result = false;

  for (int i = 0; i < DGEMM_COUNT; i++) {
    dgemms[i] = false;
  }

  parse_options(argc, argv, dgemms, &length, &random, &show_result);

  check_avx(dgemms);

  double *a = aligned_alloc(ALIGN, length * length * sizeof(double));
  double *b = aligned_alloc(ALIGN, length * length * sizeof(double));
  double *c = aligned_alloc(ALIGN, length * length * sizeof(double));

  generate_matrices(length, a, b, random);

  for (int i = 0; i < DGEMM_COUNT; i++) {
    if (dgemms[i]) {
      clean_matrix(length, c);

      clock_t start = clock(), diff;
      multiply(i, length, a, b, c);
      diff = clock() - start;

      if (show_result)
        print_matrix(length, c);

      print_result(i, length, diff);
    }
  }

  free(a);
  free(b);
  free(c);

  return 0;
}
