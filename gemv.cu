#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>

constexpr int WARP  = 32;
constexpr int BLOCK_SIZE = 256;
constexpr unsigned FULL = 0xffffffff;

__global__ void gemv_fp32_kernel(
	const float *__restrict__ A,
	const float *__restrict__ x,
	float *__restrict__ y,
	uint64_t M,
	uint64_t N
) {
	int row = blockIdx.x;
	if (row >= M) return;
	const float* a_row = A + row * N;

	float sum = 0.f;
	for (int col = threadIdx.x; col < N; col += blockDim.x)
		sum += a_row[col] * x[col];

	// warp reduce
	for (int off = WARP/2; off > 0; off >>= 1)
		sum += __shfl_down_sync(FULL, sum, off);

	__shared__ float buf[BLOCK_SIZE/WARP];
	if ((threadIdx.x & (WARP-1)) == 0)
		buf[threadIdx.x / WARP] = sum;
	__syncthreads();

	if (threadIdx.x < WARP) {
		float v = (threadIdx.x < blockDim.x/WARP) ? buf[threadIdx.x] : 0.f;
		for (int off = WARP/2; off > 0; off >>= 1)
			v += __shfl_down_sync(FULL, v, off);
		if (threadIdx.x == 0) {
			y[row] = v;
		}
	}
}

void
wrapper(const float *A, const float *x, float *y, uint64_t M, uint64_t N) {
	dim3 grid(M);
	dim3 block(BLOCK_SIZE);
	size_t shmem = (BLOCK_SIZE / WARP) * sizeof(float);
	gemv_fp32_kernel<<<grid, block, 0>>>(
		A,
		x,
		y,
		M,
		N
	);
}

void
cpu_gemv(float *A, float *x, float *y, uint64_t M, uint64_t N) {
	for (int row = 0; row < M; row++) {
		float sum = 0;
		for (int col = 0; col < N; col++) {
			sum += A[row * N + col] * x[col];
		}
		y[row] = sum;
	}
}

int main() {
	uint64_t M = 40000;
	uint64_t N = 50000;
	
	size_t mi_size = M * N * sizeof(float);
	size_t vi_size = N * sizeof(float);
	size_t vo_size = M * sizeof(float);
	
	float *hmi = (float *)malloc(M * N * sizeof(float));
	float *hvi = (float *)malloc(N * sizeof(float));
	float *hvo = (float *)malloc(M * sizeof(float));
	
	float *dmi;
	float *dvi;
	float *dvo;
	
	for (uint64_t i = 0; i < M * N; i++) {
		//hmi[i] = rand() % 1000 / 100.0;
		hmi[i] = i;
		if (i < N) {hvi[i] = i;}
	}
	
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
	
	cudaMalloc((void **)&dmi, mi_size);
	cudaMalloc((void **)&dvi, vi_size);
	cudaMalloc((void **)&dvo, vo_size);
	
	cudaMemcpy(dmi, hmi, mi_size, cudaMemcpyHostToDevice);
	cudaMemcpy(dvi, hvi, vi_size, cudaMemcpyHostToDevice);

	wrapper(dmi, dvi, dvo, M, N);

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	
	float ms = 0;
	cudaEventElapsedTime(&ms, start, stop);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	
	cudaMemcpy(hvo, dvo, vo_size, cudaMemcpyDeviceToHost);
	
	//printf("hmi:");
	//for (int i = 0; i < M * N; i++) {
	//	if (i % M == 0) {
	//		printf("\n");
	//	}
	//	printf("%2.2f ", hmi[i]);
	//}
	//printf("\n\nhvi: ");
	//for (int i = 0; i < N; i++) {
	//	printf("%2.2f ", hvi[i]);
	//}
	printf("\n\nhvo: ");
	for (int i = 0; i < M && i < 20; i++) {
		printf("%2.2f ", hvo[i]);
	}
	printf("\n");

	printf("\nGPU kernel time: %f ms\n", ms);
	
	cudaFree(dmi);
	cudaFree(dvi);
	cudaFree(dvo);
	
	//cpu_gemv(hmi, hvi, hvo, M, N);
	
	return 0;
}
