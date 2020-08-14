#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>

#define DT 2.624453295839119f  // delta t (ms)
#define DX 15.0f // delta x
#define DY 15.0f // delta y
#define V 1.500f // wave velocity v = 1.5 km/s
#define HALF_LENGTH 1 // radius of the stencil

void save_grid(int iteration, int rows, int cols, float **matrix){

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
    float **prev_base = malloc(rows * sizeof(float *));
    float **next_base = malloc(rows * sizeof(float *));

    // represent the matrix of velocities as an array
    float **vel_base = malloc(rows * sizeof(float *));

    for(int i = 0; i < rows; i++){
        prev_base[i] = malloc(cols * sizeof(float));
        next_base[i] = malloc(cols * sizeof(float));
        vel_base[i] = malloc(cols * sizeof(float));
    }

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

    float **swap;
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
//   #pragma acc enter data copyin(prev_base[:rows][:cols], next_base[:rows][:cols], vel_base[:rows][:cols]) 
    for(int n = 0; n < iterations; n++) {
	#pragma acc parallel loop collapse(2) device_type(nvidia) tile(32,32)
//	#pragma acc parallel loop collapse(2) device_type(nvidia) tile(1,1024)
//	#pragma acc parallel loop collapse(2) device_type(nvidia) tile(1,1024)
//	#pragma acc parallel loop collapse(2) device_type(nvidia) tile(32,32)
//	#pragma acc parallel loop collapse(2) device_type(nvidia) tile(64,64)
//	#pragma acc parallel loop collapse(2) device_type(nvidia) tile(128,128)
//	#pragma acc parallel loop collapse(2) device_type(nvidia) tile(256,256)
//	#pragma acc parallel loop collapse(2) device_type(nvidia) tile(512,512)
//	#pragma acc parallel loop collapse(2) device_type(nvidia) tile(1024,1024)
//	#pragma acc parallel loop device_type(nvidia) tile(32,32)
//	#pragma acc parallel loop device_type(nvidia) tile(64,64)
//	#pragma acc parallel loop device_type(nvidia) tile(128,128)
//	#pragma acc parallel loop device_type(nvidia) tile(256,256)
//	#pragma acc parallel loop device_type(nvidia) tile(512,512)
//	#pragma acc parallel loop device_type(nvidia) tile(1024,1024)
//            #pragma acc parallel loop gang
    	    for(int i = 1; i < rows - HALF_LENGTH; i++) {
		    //            #pragma acc parallel loop vector
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
        swap = next_base;
        next_base = prev_base;
        prev_base = swap;

        //if( n % 10 == 0 )
        ////    save_grid(n, rows, cols, next_base);
        //    save_grid(n, rows, cols, prev_base);
    }

    #pragma acc exit data copyout(next_base[:rows][:cols])

    // get the end time
    gettimeofday(&time_end, NULL);

    double exec_time = (double) (time_end.tv_sec - time_start.tv_sec) + (double) (time_end.tv_usec - time_start.tv_usec) / 1000000.0;

    save_grid(iterations, rows, cols, next_base);

    printf("Iterations completed in %f seconds \n", exec_time);

    return 0;
}
