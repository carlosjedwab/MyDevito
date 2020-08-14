#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>
#include <assert.h>

#define DT 0.002f // delta t
#define DX 20.0f // delta x
#define DY 20.0f // delta y
#define V 1500.0f // wave velocity v = 1500 m/s
#define HALF_LENGTH 1 // radius of the stencil
#define LENGTH 2 // radius of the stencil
#define DIMX 128
#define DIMY 1
#define NWARPS DIMX >> 5


/*
 * save the matrix on a file.txt
 */
void save_grid(int iteration, int rows, int cols, float *matrix, bool isSeq){

    system("mkdir -p wavefield");

    char file_name[256];
    sprintf(file_name, "wavefield/wavefield-iter-%d-grid-%d-%d-%d.txt", iteration, rows, cols,isSeq);

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

inline cudaError_t checkCuda(cudaError_t result)
{
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
  return result;
}

__global__ void kernelIso2dSharedMem_falido(float* next_base, float* prev_base, float* vel_base, int rows, int cols,
							float dxSquared, float dySquared, float dtSquared){

	int j = blockIdx.x * blockDim.x + threadIdx.x + 1;
	int i = blockIdx.y * blockDim.y + threadIdx.y + 1;

	int hasCompute =  (i < (rows - 1)) && (j < (cols - 1));

	__shared__ int sharedMatrix[32+HALF_LENGTH*2][32+HALF_LENGTH*2];

	 int current = i * cols + j;

	sharedMatrix[threadIdx.x+1][threadIdx.y+1] = prev_base[current];

	if(hasCompute){
		if(threadIdx.y == 0){
			sharedMatrix[threadIdx.x][threadIdx.y+1] = prev_base[current - cols];

			if(threadIdx.x == 0)
				sharedMatrix[threadIdx.x][threadIdx.y] = prev_base[current - cols -1];
			else if(threadIdx.x == 31)
				sharedMatrix[threadIdx.x+2][threadIdx.y] = prev_base[current - cols + 1];
		}
		else if(threadIdx.y == 31){
			sharedMatrix[threadIdx.x+1][threadIdx.y+1] = prev_base[current + cols];

			if(threadIdx.x == 0)
				sharedMatrix[threadIdx.x][threadIdx.y+2] = prev_base[current + cols - 1];
			else if(threadIdx.x == 31)
				sharedMatrix[threadIdx.x+2][threadIdx.y+2] = prev_base[current + cols + 1];
		}

		if((threadIdx.x == 0)){
			sharedMatrix[threadIdx.x][threadIdx.y+1] = prev_base[current - 1];
		}
		else if((threadIdx.x == 31)){
			sharedMatrix[threadIdx.x][threadIdx.y+1] = prev_base[current + 1];
		}
	}

	__syncthreads();

    // index of the current point in the grid
    int offset_x = threadIdx.x+1;
    int offset_y = threadIdx.y+1;

    // stencil code to update grid
    if(hasCompute){
		float value = 0.0;
		//neighbors in the horizontal direction
		value += (sharedMatrix[offset_x][offset_y+1] - 2.0 * sharedMatrix[offset_x][offset_y]  + sharedMatrix[offset_x][offset_y-1] ) / dxSquared;

		//neighbors in the vertical direction
		value += (sharedMatrix[offset_x+1][offset_y] - 2.0 * sharedMatrix[offset_x][offset_y] + sharedMatrix[offset_x-1][offset_y]) / dySquared;

		value *= dtSquared * vel_base[current];
		next_base[current] = 2.0 * sharedMatrix[offset_x][offset_y] - next_base[current] + value;
    }
}



__global__ void kernelIso2d_v2_falido(float* next_base, float* prev_base, float* vel_base, int rows, int cols,
							float dxSquared, float dySquared, float dtSquared){

	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;

	int warpIndex = threadIdx.x >> 5;
	int isInitialThread = (threadIdx.x - (warpIndex << 5) == 0);
	int isLastThread = (threadIdx.x - (warpIndex << 5) == 31);
	int hasCompute =  (i < (rows-1)) && (j < (cols-1));

	int current = i * cols + j;
	float value = 0.0;

	float curr = prev_base[current];

	float prevLocal=0;
	float nextLocal=0;

	float upLocal=0;
	float downLocal=0;

	__shared__ float limits[32][32];

	limits[threadIdx.y][threadIdx.x] = curr;

	__syncthreads();

	prevLocal = __shfl_sync(0xffffffff, curr, threadIdx.x-1);

	if(isInitialThread && hasCompute)
		prevLocal =  prev_base[current - 1];

	nextLocal = __shfl_sync(0xffffffff, curr, threadIdx.x-1);

	if(isLastThread && hasCompute)
		nextLocal =  prev_base[current + 1];

	if(hasCompute)
		value += (prevLocal - 2.0 * curr + nextLocal) /  dxSquared;

	if(! hasCompute)
		return;

	if(threadIdx.y == 0){
		upLocal = prev_base[current - cols];
	}else{
		upLocal = limits[threadIdx.y-1][threadIdx.x];
	}

	if(threadIdx.y == (blockDim.y - 1) ){
		downLocal = prev_base[current + cols];
	}else{
		downLocal = limits[threadIdx.y+1][threadIdx.x];
	}

	value += (upLocal  - 2.0 * curr + downLocal) / dySquared;

	value *= dtSquared * vel_base[current];
	next_base[current] = 2.0 * curr - next_base[current] + value;
}

__global__ void kernelIso2d_v4(float* next_base, const float* __restrict__ prev_base,
						       const float* __restrict__ vel_base, int rows, int cols,
						   	   const float  dxSquared, const float  dySquared, const float  dtSquared,
						   	    const int spanElementsX){

	int j = blockIdx.x * blockDim.x + threadIdx.x + HALF_LENGTH;
	int i = blockIdx.y * blockDim.y + threadIdx.y + HALF_LENGTH;

	int current = i * cols + j;
	int sPos = threadIdx.x + HALF_LENGTH;

	float valueCur;
	float values_y[LENGTH];

	int hasCompute =  (i < (rows - LENGTH )) && (j < (cols - LENGTH));
	float velBaseCurr;// = __ldg(&vel_base[current]) * dtSquared;

	__shared__ float line[DIMX + LENGTH];

	float nextBase;

	if(hasCompute){
		#pragma unroll
		for (int p = 0; p < HALF_LENGTH; ++p) {
			values_y[p] =  prev_base[ current - cols ];
			values_y[p + HALF_LENGTH] =  prev_base[ current + cols ];
		}
		nextBase = next_base[current];
		velBaseCurr = __ldg(&vel_base[current]) * dtSquared;
	}

	line[sPos - HALF_LENGTH] = __ldg( &prev_base[current - HALF_LENGTH]);

	if( (j + HALF_LENGTH < cols) && ( (threadIdx.x + LENGTH) >= blockDim.x) ){
		line[threadIdx.x + LENGTH] = __ldg( &prev_base[current + HALF_LENGTH]);
	}

	__syncthreads();

    // stencil code to update grid

	valueCur = line[sPos];

	float value = 0.0;

	float tempPrev =  __shfl_up_sync(0xffffffff, valueCur, 1);
	if( (threadIdx.x & 0x1f) == 0){
		tempPrev = line[sPos-1];
	}

//	float tempNext = line[sPos+1];
	float tempNext = __shfl_down_sync(0xffffffff, valueCur, 1);
	if( ( (threadIdx.x & 0x1f) == 31) || (threadIdx.x == blockIdx.x - spanElementsX)){
		tempNext = line[sPos + 1];
	}

	valueCur = valueCur * 2.0;

	if(!hasCompute){
		return;
	}

	value += (tempNext - valueCur + tempPrev) /  dxSquared;
	value += (values_y[0] - valueCur + values_y[1]) / dySquared;
	value *= velBaseCurr;

	next_base[current] = valueCur - nextBase + value;

}

__global__ void kernelIso2d_v3(float* next_base, const float* __restrict__ prev_base,
						       const float* __restrict__ vel_base, int rows, int cols,
						   	   const float  dxSquared, const float  dySquared, const float  dtSquared,
						   	    const int spanElementsX){

	int j = blockIdx.x * blockDim.x + threadIdx.x + HALF_LENGTH;
	int i = blockIdx.y * blockDim.y + threadIdx.y + HALF_LENGTH;

	int current = i * cols + j;
	int sPos = threadIdx.x + HALF_LENGTH;

	float valueCur;
	float values_y[LENGTH];

	int hasCompute =  (i < (rows - LENGTH )) && (j < (cols - LENGTH));
	float velBaseCurr;// = __ldg(&vel_base[current]) * dtSquared;

	__shared__ float line[DIMX + LENGTH];

	float nextBase;

	if(hasCompute){
		#pragma unroll
		for (int p = 0; p < HALF_LENGTH; ++p) {
			values_y[p] =  prev_base[ current - cols ];
			values_y[p + HALF_LENGTH] =  prev_base[ current + cols ];
		}
		nextBase = next_base[current];
		velBaseCurr = __ldg(&vel_base[current]) * dtSquared;
	}

	line[sPos - HALF_LENGTH] = __ldg( &prev_base[current - HALF_LENGTH]);

	if( (j + HALF_LENGTH < cols) && ( (threadIdx.x + LENGTH) >= blockDim.x) ){
		line[threadIdx.x + LENGTH] = __ldg( &prev_base[current + HALF_LENGTH]);
	}

	__syncthreads();

    valueCur = line[sPos] * 2.0;

	float value = 0.0;

	if(!hasCompute){
		return;
	}

	value += (line[sPos-1] - valueCur + line[sPos+1]) /  dxSquared;
	value += (values_y[0] - valueCur + values_y[1]) / dySquared;
	value *= velBaseCurr;

	next_base[current] = valueCur - nextBase + value;

}

__global__ void kernelIso2d_v1(float* next_base, float* prev_base, float* vel_base, int rows, int cols,
							float dxSquared, float dySquared, float dtSquared){

	int j = blockIdx.x * blockDim.x + threadIdx.x + HALF_LENGTH;
	int i = blockIdx.y * blockDim.y + threadIdx.y + HALF_LENGTH;

	int current = i * cols + j;


	int hasCompute =  (i < (rows - 1)) && (j < (cols - 1));

    if(hasCompute){
		int value = 0.0;

		value += (prev_base[current + 1] - 2.0 * prev_base[current] + prev_base[current - 1]) / dxSquared;
		value += (prev_base[current + cols] - 2.0 * prev_base[current] + prev_base[current - cols]) / dySquared;
		value *= dtSquared * vel_base[current];
		next_base[current] = 2.0 * prev_base[current] - next_base[current] + value;
    }
}


__global__ void kernelIso2d_v1_1(float* next_base, const float* __restrict__ prev_base,
						   const float* __restrict__ vel_base, int rows, int cols,
						   	const float  dxSquared, const float dySquared, const float  dtSquared){

	int j = blockIdx.x * blockDim.x + threadIdx.x + HALF_LENGTH;
	int i = blockIdx.y * blockDim.y + threadIdx.y + HALF_LENGTH;

	int current = i * cols + j;

	int hasCompute =  (i < (rows - 1 )) && (j < (cols - 1));

	if(!hasCompute){
		return;
	}

	float currD = __ldg(&prev_base[current]) * 2.0;
	float velBaseCurr = __ldg(&vel_base[current]) * dtSquared;

	float v1 = (__ldg(&prev_base[current + 1]) - currD + __ldg(&prev_base[current - 1])) ;
	float v2 = (__ldg(&prev_base[current + cols]) - currD + __ldg(&prev_base[current - cols]));

    // stencil code to update grid

	float value = 0.0;
	value += v1 /  dxSquared;
	value += v2 / dySquared;
	value *= velBaseCurr;

	next_base[current] = currD - next_base[current] + value;

}

double iso2dSequencial(int rows,int cols,int iterations){


    // represent the matrix of wavefield as an array
    float *prev_base = (float*) malloc(rows * cols * sizeof(float));
    float *next_base = (float*)  malloc(rows * cols * sizeof(float));

    // represent the matrix of velocities as an array
    float *vel_base = (float*) malloc(rows * cols * sizeof(float));

    // ************* BEGIN INITIALIZATION *************

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
        for(int i = 1; i < rows - HALF_LENGTH; i++) {
            for(int j = 1; j < cols - HALF_LENGTH; j++) {
                // index of the current point in the grid
                current = i * cols + j;

                // stencil code to update grid
                value = 0.0;
                //neighbors in the horizontal direction
                value += (prev_base[current + 1] - 2.0 * prev_base[current] + prev_base[current - 1]) / dxSquared;
                //neighbors in the vertical direction
                value += (prev_base[current + cols] - 2.0 * prev_base[current] + prev_base[current - cols]) / dySquared;
                value *= dtSquared * vel_base[current];
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

    save_grid(iterations, rows, cols, next_base, 1);

    printf("Sequential Algorithm - Iterations completed in %f seconds \n\n\n", exec_time);

    free(prev_base);
    free(next_base);
    free(vel_base);


    return exec_time;
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

    double sequencialTime = iso2dSequencial(rows,cols,iterations);

    // represent the matrix of wavefield as an array
    float *prev_base = (float*) malloc(rows * cols * sizeof(float));
    float *next_base = (float*) malloc(rows * cols * sizeof(float));

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

    float dxSquared = DX * DX;
    float dySquared = DY * DY;
    float dtSquared = DT * DT;

    //---------Begin - Managing structures on the GPU
    float *dv_prev_base;
    float *dv_next_base;
    float *dv_vel_base;

    cudaError_t resultCudaSuccess;
    cudaEvent_t start, stop;

//	cudaStream_t* streams  = (cudaStream_t*) malloc(iterations * sizeof(cudaStream_t));


    dim3 dimBlock(DIMX, DIMY);
//    dim3 dimGrid( ceilf( (rows)/(float) dimBlock.x), ceilf((cols)/(float) dimBlock.y) );
    dim3 dimGrid( ceilf( (cols-LENGTH)/(float) dimBlock.x), ceilf((rows-LENGTH)/(float) dimBlock.y) );
    int spanElementsX =  (cols-LENGTH) % dimBlock.x;

    printf("#block threads: %d %d | Total: %d\n",dimBlock.x,dimBlock.y, dimBlock.x*dimBlock.y);
    printf("#blocks: %d %d | Total: %d\n",dimGrid.x,dimGrid.y ,dimGrid.x*dimGrid.y);

    checkCuda( cudaEventCreate(&start) );
    checkCuda( cudaEventCreate(&stop) );
    cudaEventRecord(start);

    //Memory Allocation on the GPU
    checkCuda( cudaMalloc(&dv_prev_base, rows * cols * sizeof(float)));
    checkCuda( cudaMalloc(&dv_next_base, rows * cols * sizeof(float)));
    checkCuda( cudaMalloc(&dv_vel_base, rows * cols * sizeof(float)));

	//Synchronous transfers from CPU to GPU
//    checkCuda( cudaMemcpy(dv_next_base, next_base, rows * cols * sizeof(float), cudaMemcpyHostToDevice));
//    cudaMemset((void**)&dv_next_base, 0, rows * cols * sizeof(float));
    checkCuda( cudaMemset(dv_next_base, 0, rows * cols * sizeof(float)) );

    checkCuda( cudaMemcpy(dv_prev_base, prev_base, rows * cols * sizeof(float), cudaMemcpyHostToDevice));
    checkCuda( cudaMemcpy(dv_vel_base, vel_base, rows * cols * sizeof(float), cudaMemcpyHostToDevice));


    for(int n = 0; n < 1000; n++) {
//    	cudaStreamCreate(&streams[n]);
    	kernelIso2d_v1_1<<<dimGrid, dimBlock>>>(dv_next_base, dv_prev_base, dv_vel_base,
    											rows, cols, dxSquared, dySquared, dtSquared);

        // swap arrays for next iteration
        swap = dv_next_base;
        dv_next_base = dv_prev_base;
        dv_prev_base = swap;

//		resultCudaSuccess = cudaDeviceSynchronize();
////
//		if (resultCudaSuccess != cudaSuccess) {
//			fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(resultCudaSuccess));
//			assert(resultCudaSuccess == cudaSuccess);
//		}
    }

//   swap = dv_next_base;
//   dv_next_base = dv_prev_base;
//   dv_prev_base = swap;

//    cudaMemcpy ( void* dst, const void* src, size_t count, cudaMemcpyKind kind )
    checkCuda( cudaMemcpy(next_base, dv_prev_base, rows * cols * sizeof(float), cudaMemcpyDeviceToHost) );

	resultCudaSuccess = cudaEventRecord(stop) ;
    //Synchronous transfers from GPU to CPU
//    checkCuda( cudaMemcpy(prev_base, dv_prev_base, rows * cols * sizeof(float), cudaMemcpyDeviceToHost) );
//    checkCuda( cudaMemcpy(vel_base, dv_vel_base, rows * cols * sizeof(float), cudaMemcpyDeviceToHost) );

   	if (resultCudaSuccess != cudaSuccess) {
		fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(resultCudaSuccess));
		assert(resultCudaSuccess == cudaSuccess);
	 }

	resultCudaSuccess = cudaEventSynchronize(stop) ;

   	if (resultCudaSuccess != cudaSuccess) {
   		fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(resultCudaSuccess));
		assert(resultCudaSuccess == cudaSuccess);
   	}

	float elapsedTime;

	resultCudaSuccess = cudaEventElapsedTime(&elapsedTime, start, stop)  ;

   	if (resultCudaSuccess != cudaSuccess) {
   		fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(resultCudaSuccess));
		assert(resultCudaSuccess == cudaSuccess);
   	}

   	float exec_time = (float) (elapsedTime) / 1000.0;

    save_grid(iterations, rows, cols, next_base, 0);

    printf("Parallel Algorithm on GPU - Time: %f seconds \n", exec_time);
    printf("Speedup: %f\n", sequencialTime/exec_time);

    free(prev_base);
    free(next_base);
    free(vel_base);

    checkCuda( cudaFree(dv_prev_base) );
    checkCuda( cudaFree(dv_next_base) );
    checkCuda( cudaFree(dv_vel_base) );
    return 0;
}
