require PolyHok

PolyHok.defmodule PGemv do
  defk gemmk(a, b, c, m, n, k) do
    row = blockIdx.y * 16 + threadIdx.y
    col = blockIdx.x * 16 + threadIdx.x
    __shared__ as[256]
    __shared__ bs[256]
    sum = 0.0

    k0 = 0
    while k0 < k do
      if row < m && k0 + threadIdx.x < k do
        as[threadIdx.y * 16 + threadIdx.x] = a[row * k + k0 + threadIdx.x]
      else
        as[threadIdx.y * 16 + threadIdx.x] = 0.0
      end

      if col < n && k0 + threadIdx.y < k do
        bs[threadIdx.y * 16 + threadIdx.x] = b[(k0 + threadIdx.y) * n + col]
      else
        bs[threadIdx.y * 16 + threadIdx.x] = 0.0
      end

      __syncthreads();

      lk = 0
      while lk < 16 do
        sum = sum + as[threadIdx.y * 16 + lk] * bs[lk * 16 + threadIdx.x]
        lk = lk + 1
      end

      __syncthreads();

      k0 = k0 + 16
    end

    if row < m && col < n do
      c[row * n + col] = sum
    end
  end

  def gemm(a, b, m, n, k) do
    c = PolyHok.new_gnx(1, m * n, {:f, 32})

    PolyHok.spawn(&PGemv.gemmk/6,
                  {div(n + 16 - 1, 16), div(m + 16 - 1, 16), 1},
                  {16, 16, 1},
                  [a, b, c, m, n, k])

    c
  end
end

m = 10000;
n = 20000;
k = 30000;

a = PolyHok.new_nx_from_function_arg(1, m * k, {:f, 32}, fn x -> (m * k - x) / 3000.0 end)
b = PolyHok.new_nx_from_function_arg(1, k * n, {:f, 32}, fn x -> (k * n - x) / 2500.0 end)

IO.puts("Generated CPU inputs, starting timer...")

prev = System.monotonic_time()

gpu_a = PolyHok.new_gnx(a)
gpu_b = PolyHok.new_gnx(b)
gpu_c = PGemv.gemm(gpu_a, gpu_b, m, n, k)

next = System.monotonic_time()

_c = PolyHok.get_gnx(gpu_c)

IO.puts "time taken:\t#{System.convert_time_unit(next-prev,:native,:millisecond)}ms"
