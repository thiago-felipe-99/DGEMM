#ifndef DEBUG_H
#define DEBUG_H

#include <time.h>
#ifndef MAX_PRINT_LINE
#define MAX_PRINT_LINE 4097
#endif

void printMatrix(int length, double *matrix);

void printResult(int lenght, clock_t diff);

#endif
