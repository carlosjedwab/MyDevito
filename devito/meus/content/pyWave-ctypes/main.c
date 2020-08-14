#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include "rotinas.h"

#define DT 0.002f // delta t
#define DX 20.0f // delta x
#define DY 20.0f // delta y
#define V 1500.0f // wave velocity v = 1500 m/s
#define HALF_LENGTH 1 // radius of the stencil

#include "rotinas.c"

int main(int argc, char* argv[]) {

    if(argc != 4){
        printf("Usage: ./stencil N1 N2 ITERATIONS\n");
        printf("N1 N2: grid sizes for the stencil\n");
        printf("ITERATIONS: number of timesteps\n");
        exit(-1);
    }

    // number of rows of the grid
    int rows = atoi(argv[1]);

    // number of columns of the grid
    int cols = atoi(argv[2]);

    // number of timesteps
    int iterations = atoi(argv[3]);

    // represent the matrix of wavefield as an array
    float *prev_base = malloc(rows * cols * sizeof(float));
    //float *next_base = malloc(rows * cols * sizeof(float));

    // represent the matrix of velocities as an array
    float *vel_base = malloc(rows * cols * sizeof(float));

    printf("Grid Sizes: %d x %d\n", rows, cols);
    printf("Iterations: %d\n", iterations);

    // ************* BEGIN INITIALIZATION *************

    printf("Initializing ... \n");

    // define source wavelet
    float wavelet[12] = {0.016387336, -0.041464937, -0.067372555, 0.386110067,
                         0.812723635, 0.416998396,  0.076488599,  -0.059434419,
                         0.023680172, 0.005611435,  0.001823209,  -0.000720549};

    // initialize matrix
    for(int i = 0; i < rows; i++){

        int offset = i * cols;

        for(int j = 0; j < cols; j++){
            prev_base[offset + j] = 0.0f;
            //next_base[offset + j] = 0.0f;
            vel_base[offset + j] = V * V;
        }
    }

    // add a source to initial wavefield as an initial condition
    for(int s = 11; s >= 0; s--){
        for(int i = rows / 2 - s; i < rows / 2 + s; i++){

            int offset = i * cols;

            for(int j = cols / 2 - s; j < cols / 2 + s; j++)
                prev_base[offset + j] = wavelet[s];
        }
    }

    // ************** END INITIALIZATION **************

    printf("Computing wavefield ... \n");



    // variable to measure execution time
    struct timeval time_start;
    struct timeval time_end;

    // get the start time
    gettimeofday(&time_start, NULL);
    
    acoustic_forward(prev_base, vel_base, (size_t) rows, (size_t) cols, (size_t) iterations, DX, DY, DT, 100);
    
    // get the end time
    gettimeofday(&time_end, NULL);

    double exec_time = (double) (time_end.tv_sec - time_start.tv_sec) + (double) (time_end.tv_usec - time_start.tv_usec) / 1000000.0;

    //save_grid(iterations, rows, cols, next_base);

    printf("Iterations completed in %f seconds \n", exec_time);

    free(prev_base);
    //free(next_base);
    free(vel_base);

    return 0;
}
