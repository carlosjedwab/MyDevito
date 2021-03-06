#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <sys/time.h>

#define DT 0.002f // delta t
#define DX 20.0f // delta x
#define DY 20.0f // delta y
#define V 1500.0f // wave velocity v = 1500 m/s
#define HALF_LENGTH 1 // radius of the stencil

/*
 * save the matrix on a file.txt
 */
void save_grid(int iteration, int rows, int cols, float *matrix){

    system("mkdir -p wavefield");

    char file_name[256];
    sprintf(file_name, "wavefield/wavefield-iter-%d-grid-%d-%d.txt", iteration, rows, cols);

    // save the result
    FILE *file;
    file = fopen(file_name, "w");

    for(int i = 0; i < rows; i++) {

        int offset = i * cols;

        for(int j = 0; j < cols; j++) {
            fprintf(file, "%f ", matrix[offset + j]);
        }
        fprintf(file, "\n");
    }

    fclose(file);
}

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
    float *prev_base = (float*) malloc(rows * cols * sizeof(float));
    float *next_base = (float*) malloc(rows * cols * sizeof(float));
    float *density = (float*) malloc(rows * cols * sizeof(float));

    // represent the matrix of velocities as an array
    float *vel_base = (float*) malloc(rows * cols * sizeof(float));

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
            next_base[offset + j] = 0.0f;
            vel_base[offset + j] = V * V;
            density[offset + j] = 1.0f;
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

    float *swap;
    float value = 0.0;
    int current;

    float dxSquared = DX * DX;
    float dySquared = DY * DY;
    float dtSquared = DT * DT;

    // variable to measure execution time
    struct timeval time_start;
    struct timeval time_end;
    
    float x1, x2, y1, y2, term_x, term_y;
    
    // get the start time
    gettimeofday(&time_start, NULL);

    // wavefield modeling
    for(int n = 0; n < iterations; n++) {
        for(int i = 1; i < rows - HALF_LENGTH; i++) {
            for(int j = 1; j < cols - HALF_LENGTH; j++) {
                // index of the current point in the grid
                current = i * cols + j;
                
                //neighbors in the horizontal direction
                x1 = ((prev_base[current + 1] - prev_base[current]) * (density[current + 1] + density[current])) / density[current + 1];
                x2 = ((prev_base[current] - prev_base[current - 1]) * (density[current] + density[current - 1])) / density[current - 1];
                term_x = (x1 - x2) / (2 * dxSquared);
                
                //neighbors in the vertical direction
                y1 = ((prev_base[current + cols] - prev_base[current]) * (density[current + cols] + density[current])) / density[current + cols];
                y2 = ((prev_base[current] - prev_base[current - cols]) * (density[current] + density[current - cols])) / density[current - cols];
                term_y = (y1 - y2) / (2 * dySquared);  
                
                //value *= dtSquared * vel_base[current];
                value = dtSquared * vel_base[current] * (term_x + term_y);
                next_base[current] = 2.0 * prev_base[current] - next_base[current] + value;
            }
        }

        // swap arrays for next iteration
        swap = next_base;
        next_base = prev_base;
        prev_base = swap;

        //if( n % 10 == 0 )
        //    save_grid(n, rows, cols, next_base);
    }

    // get the end time
    gettimeofday(&time_end, NULL);

    double exec_time = (double) (time_end.tv_sec - time_start.tv_sec) + (double) (time_end.tv_usec - time_start.tv_usec) / 1000000.0;

    save_grid(iterations, rows, cols, next_base);

    printf("Iterations completed in %f seconds \n", exec_time);

    free(prev_base);
    free(next_base);
    free(vel_base);

    return 0;
}
