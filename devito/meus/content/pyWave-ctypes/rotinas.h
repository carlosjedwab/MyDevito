#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>

#define HALF_LENGTH 1 // radius of the stencil

/*
 * save the matrix on a file.txt
 */
void save_grid(int iteration, int rows, int cols, float *matrix);

void acoustic_forward(float *grid, float *vel_base, size_t nx, size_t ny, size_t nt, float dx, float dy, float dt, int print_every);

