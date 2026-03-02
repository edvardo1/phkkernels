require PolyHok

PolyHok.defmodule PGemv do
  defk gevmk(x, a, y, m, n) do
    col = blockIdx.x
    if col < n do
      sum = 0.0
      row = threadIdx.x
      while row < m do
        sum = sum + x[row] * a[row * n + col]
        row = row + blockDim.x
      end

      off = 16
      while off > 0 do
	    sum = sum + __shfl_down_sync(0xffffffff, sum, off)
        off = off / 2
      end

	  __shared__ buf[8]
      if (threadIdx.x / 32) * 32 == threadIdx.x do
        buf[threadIdx.x / 32] = sum
      end
      __syncthreads()

      if threadIdx.x < 32 do
        v = 0
        if threadIdx.x < blockDim.x / 32 do
          v = buf[threadIdx.x]
        end

        off = 8
        while off > 0 do
          v = v + __shfl_down_sync(0xffffffff, v, off)
          off = off / 2
        end

        if threadIdx.x == 0 do
          y[col] = v
        end
      end
    end
  end

  def gevm(a, x, m, n) do
    y = PolyHok.new_gnx(1, n, {:f, 32})

    PolyHok.spawn(&PGemv.gevmk/5,
                  {div(n + 32 - 1, 32), 1, 1},
                  {32 * 8, 1, 1},
                  [x, a, y, m, n])

    y
  end
end

m = 10000;
n = 60000;

a = PolyHok.new_nx_from_function_arg(1, m * n, {:f, 32}, fn x -> (x) / 3000.0 end)
x = PolyHok.new_nx_from_function_arg(1, m,     {:f, 32}, fn x -> (x) / 2500.0 end)

prev = System.monotonic_time()

gpu_a = PolyHok.new_gnx(a)
gpu_x = PolyHok.new_gnx(x)
gpu_y = PGemv.gevm(gpu_a, gpu_x, m, n)

next = System.monotonic_time()

_y = PolyHok.get_gnx(gpu_y)

IO.puts "time taken:\t#{System.convert_time_unit(next-prev,:native,:millisecond)}ms"
