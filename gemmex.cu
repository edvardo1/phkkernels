#include <assert.h>
#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>

constexpr int BLK  = 16;           // 16×16 tile  ⇒ 256 threads / CTA
constexpr int WARP = 32;
constexpr unsigned FULL = 0xffffffff;

//__global__ void gemm_tiled_fp32_kernel(const float* __restrict__ A,
//                                       const float* __restrict__ B,
//                                       float*       __restrict__ C,
//                                       int  M, int  N, int  K)
//{
//    const int row = blockIdx.y * BLK + threadIdx.y;
//    const int col = blockIdx.x * BLK + threadIdx.x;
//	//printf("row = %d, col = %d\n", row, col);
//
//    __shared__ float As[BLK][BLK];
//    __shared__ float Bs[BLK][BLK];
//
//    float sum = 0.f;
//
//    for (int k0 = 0; k0 < K; k0 += BLK) {
//        if (row < M && k0 + threadIdx.x < K)
//            As[threadIdx.y][threadIdx.x] = A[(size_t)row * K + (k0 + threadIdx.x)];
//        else
//            As[threadIdx.y][threadIdx.x] = 0.f;
//
//        if (col < N && k0 + threadIdx.y < K)
//            Bs[threadIdx.y][threadIdx.x] = B[(size_t)(k0 + threadIdx.y) * N + col];
//        else
//            Bs[threadIdx.y][threadIdx.x] = 0.f;
//
//        __syncthreads();
//
//#pragma unroll
//        for (int k = 0; k < BLK; ++k)
//            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
//
//        __syncthreads();
//    }
//
//    if (row < M && col < N)
//        C[(size_t)row * N + col] = sum;
//}

__global__ void gemmk(float *a, float *b, float *c, int m, int n, int k)
{
	int row = ((blockIdx.y * 16) + threadIdx.y);
	int col = ((blockIdx.x * 16) + threadIdx.x);
__shared__ float as[256];
__shared__ float bs[256];
	float sum = 0.0;
	int k0 = 0;
while((k0 < k)){
if(((row < m) && ((k0 + threadIdx.x) < k)))
{
	as[((threadIdx.y * 16) + threadIdx.x)] = a[(((row * k) + k0) + threadIdx.x)];
}
else{
	as[((threadIdx.y * 16) + threadIdx.x)] = 0.0;
}

if(((col < n) && ((k0 + threadIdx.y) < k)))
{
	bs[((threadIdx.y * 16) + threadIdx.x)] = b[(((k0 + threadIdx.y) * n) + col)];
}
else{
	bs[((threadIdx.y * 16) + threadIdx.x)] = 0.0;
}

__syncthreads();
	sum = (sum + (as[((threadIdx.y * 16) + 0)] * bs[((0 * 16) + threadIdx.x)]));
	sum = (sum + (as[((threadIdx.y * 16) + 1)] * bs[((1 * 16) + threadIdx.x)]));
	sum = (sum + (as[((threadIdx.y * 16) + 2)] * bs[((2 * 16) + threadIdx.x)]));
	sum = (sum + (as[((threadIdx.y * 16) + 3)] * bs[((3 * 16) + threadIdx.x)]));
	sum = (sum + (as[((threadIdx.y * 16) + 4)] * bs[((4 * 16) + threadIdx.x)]));
	sum = (sum + (as[((threadIdx.y * 16) + 5)] * bs[((5 * 16) + threadIdx.x)]));
	sum = (sum + (as[((threadIdx.y * 16) + 6)] * bs[((6 * 16) + threadIdx.x)]));
	sum = (sum + (as[((threadIdx.y * 16) + 7)] * bs[((7 * 16) + threadIdx.x)]));
	sum = (sum + (as[((threadIdx.y * 16) + 8)] * bs[((8 * 16) + threadIdx.x)]));
	sum = (sum + (as[((threadIdx.y * 16) + 9)] * bs[((9 * 16) + threadIdx.x)]));
	sum = (sum + (as[((threadIdx.y * 16) + 10)] * bs[((10 * 16) + threadIdx.x)]));
	sum = (sum + (as[((threadIdx.y * 16) + 11)] * bs[((11 * 16) + threadIdx.x)]));
	sum = (sum + (as[((threadIdx.y * 16) + 12)] * bs[((12 * 16) + threadIdx.x)]));
	sum = (sum + (as[((threadIdx.y * 16) + 13)] * bs[((13 * 16) + threadIdx.x)]));
	sum = (sum + (as[((threadIdx.y * 16) + 14)] * bs[((14 * 16) + threadIdx.x)]));
	sum = (sum + (as[((threadIdx.y * 16) + 15)] * bs[((15 * 16) + threadIdx.x)]));
__syncthreads();
	k0 = (k0 + 16);
}
if(((row < m) && (col < n)))
{
	c[((row * n) + col)] = sum;
}

}

void
wrapper(float *A, float *B, float *C, uint64_t M, uint64_t N, uint64_t K) {
	//dim3 block(BLK, BLK);
	//dim3 grid((N+BLK-1)/BLK, (M+BLK-1)/BLK);
	dim3 block(BLK, BLK);
	dim3 grid((N+BLK-1)/BLK,
	          (M+BLK-1)/BLK);

	gemmk<<<grid, block>>>(A, B, C, M, N, K);
}

int main() {
	//const int M = 100;
	//const int N = 200;
	//const int K = 300;
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
	
	//for (uint64_t i = 0; i < M * K; i++) {ha[i] = i / 3000.0;}
	//for (uint64_t i = 0; i < K * N; i++) {hb[i] = i / 2500.0;}
	for (uint64_t i = 0; i < M * K; i++) {ha[i] = i / 3000.0;}
	for (uint64_t i = 0; i < K * N; i++) {hb[i] = i / 2500.0;}

	printf("\n\nha [%d]: ", (uint64_t)a_size / (uint64_t)sizeof(float));
	for (uint64_t i = 0; i < M * K && i < 20; i++) {
		printf("%f ", ha[i]);
	}
	printf("\n\nhb [%d]: ", (uint64_t)b_size / (uint64_t)sizeof(float));
	for (uint64_t i = 0; i < K * N && i < 20; i++) {
		printf("%f ", hb[i]);
	}
	printf("\n");
	
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
	
	printf("\n\nhc [%d]: ", M * N);
	for (int i = 0; i < M * N && i < 5; i++) {
		printf("%f ", hc[i]);
	}
	printf("...");
	//for (int i = 0; i < 5; i++) {
	//	assert((M * N - 1 - 10 + i) < (M * N));
	//	printf("%f ", hc[M * N - 1 - 10 + i]);
	//}
	printf("\n");

	printf("\nGPU kernel time: %f ms\n", ms);
	
	cudaFree(da);
	cudaFree(db);
	cudaFree(dc);
	
	return 0;
}
