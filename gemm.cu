#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>

constexpr int BLK  = 16;           // 16×16 tile  ⇒ 256 threads / CTA
constexpr int WARP = 32;
constexpr unsigned FULL = 0xffffffff;

__global__ void gemm_tiled_fp32_kernel(const float* __restrict__ A,
                                       const float* __restrict__ B,
                                       float*       __restrict__ C,
                                       int  M, int  N, int  K)
{
    const int row = blockIdx.y * BLK + threadIdx.y;
    const int col = blockIdx.x * BLK + threadIdx.x;

    __shared__ float As[BLK][BLK];
    __shared__ float Bs[BLK][BLK];

    float sum = 0.f;

    for (int k0 = 0; k0 < K; k0 += BLK) {
        if (row < M && k0 + threadIdx.x < K)
            As[threadIdx.y][threadIdx.x] =
                A[(size_t)row * K + (k0 + threadIdx.x)];
        else
            As[threadIdx.y][threadIdx.x] = 0.f;

        if (col < N && k0 + threadIdx.y < K)
            Bs[threadIdx.y][threadIdx.x] =
                B[(size_t)(k0 + threadIdx.y) * N + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.f;

        __syncthreads();

#pragma unroll
        for (int k = 0; k < BLK; ++k)
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];

        __syncthreads();
    }

    if (row < M && col < N)
        C[(size_t)row * N + col] = sum;
}


void
wrapper(const float *A, const float *B, float *C, uint64_t M, uint64_t N, uint64_t K) {
	//dim3 block(BLK, BLK);
	//dim3 grid((N+BLK-1)/BLK, (M+BLK-1)/BLK);
	dim3 block(BLK, BLK);
	dim3 grid((N+BLK-1)/BLK,
	          (M+BLK-1)/BLK);
	//printf("{%d, %d, %d}\n", (N+BLK-1)/BLK, (M+BLK-1)/BLK, 0);

	gemm_tiled_fp32_kernel<<<grid, block>>>(A, B, C, M, N, K);
}

int main() {
	const uint64_t M = 10000;
	const uint64_t N = 20000;
	const uint64_t K = 30000;
	
	size_t a_size = M * K * sizeof(float);
	size_t b_size = K * N * sizeof(float);
	size_t c_size = M * N * sizeof(float);
	
	float *ha = (float *)malloc(a_size * sizeof(float));
	float *hb = (float *)malloc(b_size * sizeof(float));
	float *hc = (float *)malloc(c_size * sizeof(float));
	
	float *da;
	float *db;
	float *dc;
	
	for (uint64_t i = 0; i < M * K; i++) {ha[i] = i / 3000.0;}
	for (uint64_t i = 0; i < K * N; i++) {hb[i] = i / 2500.0;}
	
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
	
	cudaMalloc((void **)&da, a_size);
	cudaMalloc((void **)&db, b_size);
	cudaMalloc((void **)&dc, c_size);
	
	cudaMemcpy(da, ha, a_size, cudaMemcpyHostToDevice);
	cudaMemcpy(db, hb, b_size, cudaMemcpyHostToDevice);

	wrapper(da, db, dc, M, N, K);

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	
	float ms = 0;
	cudaEventElapsedTime(&ms, start, stop);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	
	cudaMemcpy(hc, dc, c_size, cudaMemcpyDeviceToHost);
	
	printf("\n\nhc: ");
	for (int i = 0; i < M * N && i < 20; i++) {
		printf("%2.2f ", hc[i]);
	}
	printf("\n");

	printf("\nGPU kernel time: %f ms\n", ms);
	
	cudaFree(da);
	cudaFree(db);
	cudaFree(dc);
	
	return 0;
}
