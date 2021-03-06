#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>

#define DT 0.002f // delta t
#define DX 20.0f // delta x
#define DY 20.0f // delta y
#define V 1500.0f // wave velocity v = 1500 m/s
#define HALF_LENGTH 1 // radius of the stencil

void save_grid(int iteration, int rows, int cols, float matrix[rows][cols]){

    system("mkdir -p wavefield");

    char file_name[256];
    sprintf(file_name, "wavefield/wavefield-iter-%d-grid-%d-%d.txt", iteration, rows, cols);

    // save the result
    FILE *file;
    file = fopen(file_name, "w");

    for(int i = 0; i < rows; i++) {
        for(int j = 0; j < cols; j++) {
            fprintf(file, "%f ", matrix[i][j]);
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

    // matrix used to update the wavefield
    float prev_base[rows][cols];
    float next_base[rows][cols];

    // represent the matrix of velocities as an array
    float vel_base[rows][cols];

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
        for(int j = 0; j < cols; j++){
            prev_base[i][j] = 0.0f;
            next_base[i][j] = 0.0f;
            vel_base[i][j] = V * V;
        }
    }

    // add a source to initial wavefield as an initial condition
    for(int s = 11; s >= 0; s--){
        for(int i = rows / 2 - s; i < rows / 2 + s; i++){
            for(int j = cols / 2 - s; j < cols / 2 + s; j++)
                prev_base[i][j] = wavelet[s];
        }
    }

    // ************** END INITIALIZATION **************

    printf("Computing wavefield ... \n");

    float value = 0.0;

    float dxSquared = DX * DX;
    float dySquared = DY * DY;
    float dtSquared = DT * DT;

    // variable to measure execution time
    struct timeval time_start;
    struct timeval time_end;

    // get the start time
    gettimeofday(&time_start, NULL);

    // wavefield modeling
    for(int n = 0; n < iterations; n++) {
        for(int i = 1; i < rows - HALF_LENGTH; i++) {
            for(int j = 1; j < cols - HALF_LENGTH; j++) {
                // stencil code to update grid
                value = 0.0;
                //neighbors in the horizontal direction
                value += (prev_base[i][j+1] - 2.0 * prev_base[i][j] + prev_base[i][j-1]) / dxSquared;
                //neighbors in the vertical direction
                value += (prev_base[i+1][j] - 2.0 * prev_base[i][j] + prev_base[i-1][j]) / dySquared;
                value *= dtSquared * vel_base[i][j];
                next_base[i][j] = 2.0 * prev_base[i][j] - next_base[i][j] + value;
            }
        }

        // swap arrays for next iteration
        float temp;
        for(int i = 0; i < rows; i++) {
            for(int j = 0; j < cols; j++) {
                temp = prev_base[i][j];
                prev_base[i][j] = next_base[i][j];
                next_base[i][j] = temp;
            }
        }

        //if( n % 10 == 0 )
        //    save_grid(n, rows, cols, next_base);
    }

    // get the end time
    gettimeofday(&time_end, NULL);

    double exec_time = (double) (time_end.tv_sec - time_start.tv_sec) + (double) (time_end.tv_usec - time_start.tv_usec) / 1000000.0;

    save_grid(iterations, rows, cols, next_base);

    printf("Iterations completed in %f seconds \n", exec_time);

    return 0;
}
