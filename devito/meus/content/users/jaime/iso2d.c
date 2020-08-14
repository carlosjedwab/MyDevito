#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>

#define DT 0.002f // delta t
#define DX 20.0f // delta x
#define DY 20.0f // delta y
#define V 1500.0f // wave velocity v = 1500 m/s

/*
 * save the matrix on a file.txt
 */
void save_grid(int iteration, int rows, int cols, int stencil_radius, float *matrix){

    system("mkdir -p wavefield");

    int spatial_order = stencil_radius * 2;

    char file_name[256];
    sprintf(file_name, "wavefield/wavefield-iter-%d-grid-%d-%d.txt", iteration, rows-spatial_order, cols-spatial_order);

    // save the result
    FILE *file;
    file = fopen(file_name, "w");

    for(int i = stencil_radius; i < rows - stencil_radius; i++) {

        int offset = i * cols;

        for(int j = stencil_radius; j < cols - stencil_radius; j++) {
            fprintf(file, "%f ", matrix[offset + j]);
        }
        fprintf(file, "\n");
    }

    fclose(file);
}

int main(int argc, char* argv[]) {

    // validate the parameters
    if(argc != 5){
        printf("Usage: ./stencil N1 N2 ITERATIONS SPATIAL_ORDER\n");
        printf("N1 N2: grid sizes for the stencil\n");
        printf("ITERATIONS: number of timesteps\n");
        printf("SPATIAL_ORDER: number of neighbors to calculate. Accept values are 2, 4, 6, 8, 10, 12, 14 or 16\n");
        exit(-1);
    }

    // number of rows of the grid
    int rows = atoi(argv[1]);

    // number of columns of the grid
    int cols = atoi(argv[2]);

    // number of timesteps
    int iterations = atoi(argv[3]);

    // number of timesteps
    int spatial_order = atoi(argv[4]);

    // validate the spatial order
    if( spatial_order % 2 != 0 || spatial_order < 2 || spatial_order > 16 ){
        printf("ERROR: spatial order must be 2, 4, 6, 8, 10, 12, 14 or 16\n");
        exit(-1);
    }

    // radius of the stencil
    int stencil_radius = spatial_order / 2;

    printf("Grid Sizes: %d x %d\n", rows, cols);
    printf("Iterations: %d\n", iterations);
    printf("Spatial Order: %d\n", spatial_order);

    // add the spatial order to the grid size
    rows += spatial_order;
    cols += spatial_order;

    // represent the matrix of wavefield as an array
    float *prev_base = malloc(rows * cols * sizeof(float));
    float *next_base = malloc(rows * cols * sizeof(float));

    // represent the matrix of velocities as an array
    float *vel_base = malloc(rows * cols * sizeof(float));

    // ************* BEGIN INITIALIZATION *************

    printf("Initializing ... \n");

    // array of coefficients
    float coefficient[stencil_radius + 1];

    // get the coefficients for the specific spatial ordem
    switch (spatial_order){
        case 2:
            coefficient[0] = -2.0;
            coefficient[1] = 1.0;
        break;

        case 4:
            coefficient[0] = -2.50000e+0;
            coefficient[1] = 1.33333e+0;
            coefficient[2] = -8.33333e-2;
        break;

        case 6:
            coefficient[0] = -2.72222e+0;
            coefficient[1] = 1.50000e+0;
            coefficient[2] = -1.50000e-1;
            coefficient[3] = 1.11111e-2;
        break;

        case 8:
            coefficient[0] = -2.84722e+0;
            coefficient[1] = 1.60000e+0;
            coefficient[2] = -2.00000e-1;
            coefficient[3] = 2.53968e-2;
            coefficient[4] = -1.78571e-3;
        break;

        case 10:
            coefficient[0] = -2.92722e+0;
            coefficient[1] = 1.66667e+0;
            coefficient[2] = -2.38095e-1;
            coefficient[3] = 3.96825e-2;
            coefficient[4] = -4.96032e-3;
            coefficient[5] = 3.17460e-4;
        break;

        case 12:
            coefficient[0] = -2.98278e+0;
            coefficient[1] = 1.71429e+0;
            coefficient[2] = -2.67857e-1;
            coefficient[3] = 5.29101e-2;
            coefficient[4] = -8.92857e-3;
            coefficient[5] = 1.03896e-3;
            coefficient[6] = -6.01251e-5;
        break;

        case 14:
            coefficient[0] = -3.02359e+0;
            coefficient[1] = 1.75000e+0;
            coefficient[2] = -2.91667e-1;
            coefficient[3] = 6.48148e-2;
            coefficient[4] = -1.32576e-2;
            coefficient[5] = 2.12121e-3;
            coefficient[6] = -2.26625e-4;
            coefficient[7] = 1.18929e-5;
        break;

        case 16:
            coefficient[0] = -3.05484e+0;
            coefficient[1] = 1.77778e+0;
            coefficient[2] = -3.11111e-1;
            coefficient[3] = 7.54209e-2;
            coefficient[4] = -1.76768e-2;
            coefficient[5] = 3.48096e-3;
            coefficient[6] = -5.18001e-4;
            coefficient[7] = 5.07429e-5;
            coefficient[8] = -2.42813e-6;
        break;
    }

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

    // get the start time
    gettimeofday(&time_start, NULL);

    // wavefield modeling
    for(int n = 0; n < iterations; n++) {
        for(int i = stencil_radius; i < rows - stencil_radius; i++) {
            for(int j = stencil_radius; j < cols - stencil_radius; j++) {
                // index of the current point in the grid
                current = i * cols + j;

                // stencil code to update grid
                value = 0.0;
                value += coefficient[0] * (prev_base[current]/dxSquared + prev_base[current]/dySquared);

                // radius of the stencil
                for(int ir = 1; ir <= stencil_radius; ir++){
                    value += coefficient[ir] * (
                            ( (prev_base[current + 1] + prev_base[current - 1]) / dxSquared ) + //neighbors in the horizontal direction
                            ( (prev_base[current + cols] + prev_base[current - cols]) / dySquared )); //neighbors in the vertical direction
                }

                value *= dtSquared * vel_base[current];
                next_base[current] = 2.0 * prev_base[current] - next_base[current] + value;
            }
        }

        // swap arrays for next iteration
        swap = next_base;
        next_base = prev_base;
        prev_base = swap;

        //if( n % 10 == 0 )
        //    save_grid(n, rows, cols, stencil_radius, next_base);
    }

    // get the end time
    gettimeofday(&time_end, NULL);

    double exec_time = (double) (time_end.tv_sec - time_start.tv_sec) + (double) (time_end.tv_usec - time_start.tv_usec) / 1000000.0;

    save_grid(iterations, rows, cols, stencil_radius, next_base);

    printf("Iterations completed in %f seconds \n", exec_time);

    free(prev_base);
    free(next_base);
    free(vel_base);

    return 0;
}
