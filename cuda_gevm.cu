
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <cuda.h>
#include <cuda_runtime.h>

constexpr int WARP       = 32;
constexpr int WARPS_PER_BLK  = 8;
constexpr int BLOCK_SIZE     = WARP * WARPS_PER_BLK;
constexpr unsigned FULL  = 0xffffffff;

//__global__ void gevm_fp32_kernel(
//	const float* __restrict__ x,
//	const float* __restrict__ A,
//	float*       __restrict__ y,
//	int M, int N
//) {
//	int col = blockIdx.x;
//	if (col >= N) return;
//
//	float sum = 0.f;
//	for (int row = threadIdx.x; row < M; row += blockDim.x)
//		sum += x[row] * A[row * N + col];
//
//	for (int off = WARP / 2; off > 0; off >>= 1)
//		sum += __shfl_down_sync(FULL, sum, off);
//
//	__shared__ float buf[BLOCK_SIZE / WARP];
//	if ((threadIdx.x & (WARP - 1)) == 0)
//		buf[threadIdx.x / WARP] = sum;
//	__syncthreads();
//
//	if (threadIdx.x < WARP) {
//		float v = (threadIdx.x < blockDim.x / WARP) ? buf[threadIdx.x] : 0.f;
//		for (int off = WARP / 2; off > 0; off >>= 1)
//			v += __shfl_down_sync(FULL, v, off);
//		if (threadIdx.x == 0)
//			y[col] = v;
//	}
//}

__global__ void gevm_tiled_fp32_kernel(const float* __restrict__ x,
                                       const float* __restrict__ A,
                                       float*       __restrict__ y,
                                       int M, int N)
{
    const int lane   = threadIdx.x;               // 0…31
    const int warpId = threadIdx.y;               // 0…7
    const int colBase= blockIdx.x * WARP;
    const int col    = colBase + lane;

    float sum = 0.f;

    for (int row = warpId; row < M; row += WARPS_PER_BLK)
    {
        float xv = 0.f;
        if (lane == 0)
            xv = x[row];
        xv = __shfl_sync(FULL, xv, 0);

        float a = 0.f;
        if (col < N)
            a = A[row * N + col];

        sum += xv * a;
    }

    __shared__ float sm[WARPS_PER_BLK][WARP];     // 1 KB
    sm[warpId][lane] = sum;
    __syncthreads();

    if (warpId == 0 && col < N)
    {
        float colSum = 0.f;
        #pragma unroll
        for (int w = 0; w < WARPS_PER_BLK; ++w)
            colSum += sm[w][lane];
        y[col] = colSum;
    }
}

void wrapper(
	float *x,
	float *A,
	float *y,
	uint64_t M,
	uint64_t N
) {
	dim3 grid((N + WARP - 1) / WARP);
	dim3 block(WARP, 8);
	gevm_tiled_fp32_kernel<<<grid, block>>>(x, A, y, M, N);
}

int main() {
	assert(BLOCK_SIZE / WARP == 8);
	const uint64_t M = finput_m;
	const uint64_t N = finput_n;
	
	size_t A_size = M * N * sizeof(float);
	size_t x_size = M * sizeof(float);
	size_t y_size = N * sizeof(float);
	
	float *hA = (float *)malloc(A_size);
	float *hx = (float *)malloc(x_size);
	float *hy = (float *)malloc(y_size);
	
	float *dA = NULL;
	float *dx = NULL;
	float *dy = NULL;
	
	for (uint64_t i = 0; i < M * N; i++) {hA[i] = 2;}
	for (uint64_t i = 0; i < M; i++)     {hx[i] = 2;}
	
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	//cudaEventRecord(start);
	cudaEventRecord(start);
	
	if (cudaMalloc((void **)&dA, A_size) != cudaSuccess) {
		assert(0);
	}
	if (cudaMalloc((void **)&dx, x_size) != cudaSuccess) {
		assert(0);
	}
	if (cudaMalloc((void **)&dy, y_size) != cudaSuccess) {
		assert(0);
	}

	cudaMemcpy(dA, hA, A_size, cudaMemcpyHostToDevice);
	cudaMemcpy(dx, hx, x_size, cudaMemcpyHostToDevice);

	wrapper(dx, dA, dy, M, N);
	
	cudaMemcpy(hy, dy, y_size, cudaMemcpyDeviceToHost);

	cudaDeviceSynchronize();

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	
	float ms = 0;
	cudaEventElapsedTime(&ms, start, stop);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	if (finput_print_result) {
		printf("hx = [");
		for (int i = 0; i < 5; i++) {
			if (i != 0) {
				printf(", ");
			}
			printf("%f", hx[i]);
		}
		printf("...]\n");

		printf("hA = [");
		for (int i = 0; i < 5; i++) {
			if (i != 0) {
				printf(", ");
			}
			printf("%f", hA[i]);
		}
		printf("...]\n");

		printf("hy = [");
		for (int i = 0; i < 5; i++) {
			if (i != 0) {
				printf(", ");
			}
			printf("%f", hy[i]);
		}
		printf("...]\n");
	}

	printf("cuda: %fms\n", ms);
	
	cudaFree(dA);
	cudaFree(dx);
	cudaFree(dy);
	
	return 0;

}
