require PolyHok

PolyHok.defmodule PGemv do
  #defk gevmk(x, a, y, m, n) do
  #  col = blockIdx.x
  #  if col < n do
  #    sum = 0.0
  #    row = threadIdx.x
  #    while row < m do
  #      sum = sum + x[row] * a[row * n + col]
  #      row = row + blockDim.x
  #    end
  #
  #    off = 16
  #    while off > 0 do
  #      sum = sum + __shfl_down_sync(0xffffffff, sum, off)
  #      off = off / 2
  #    end
  #
  #    __shared__ buf[8]
  #    if (threadIdx.x / 32) * 32 == threadIdx.x do
  #      buf[threadIdx.x / 32] = sum
  #    end
  #    __syncthreads()
  #
  #    if threadIdx.x < 32 do
  #      v = 0.0
  #      if threadIdx.x < blockDim.x / 32 do
  #        v = buf[threadIdx.x]
  #      end
  #
  #      off = 8
  #      while off > 0 do
  #        v = v + __shfl_down_sync(0xffffffff, v, off)
  #        off = off / 2
  #      end
  #
  #      if threadIdx.x == 0 do
  #        y[col] = v
  #      end
  #    end
  #  end
  #end

  defk gevmk(x, a, y, m, n) do
    lane    = threadIdx.x
    warpId  = threadIdx.y
    colbase = blockIdx.x * 32
    col = colbase + lane

    sum = 0.0

    row = warpId
    while row < m do
      xv = 0.0
      if lane == 0 do
        xv = x[row]
      end
      xv = __shfl_sync(0xffffffff, xv, 0)

      sa = 0.0
      if col < n do
        sa = a[row * n + col]
      end

      sum = sum + xv * sa 

      row = row + 8
    end

    __shared__ sm[8 * 32]
    #sm[warpId * 32 + lane] = sum
    sm[warpId * lane] = sum
    __syncthreads()

    if warpId == 0 do
      if col < n do
        colsum = 0.0
        w = 0
        while w < 8 do
          colsum = colsum + sm[w * warpId + lane]
          w = w + 1
        end
        y[col] = colsum
      end
    end
  end

  def gevm(a, x, m, n) do
    y = PolyHok.new_gnx(1, n, {:f, 32})

    PolyHok.spawn(&PGemv.gevmk/5,
                  {div(n + 32 - 1, 32), 1, 1},
                  {32, 8, 1},
                  [x, a, y, m, n])

    y
  end
end

m = finput_m;
n = finput_n;

a = PolyHok.new_nx_from_function_arg(1, m * n, {:f, 32}, fn x -> 2 end)
x = PolyHok.new_nx_from_function_arg(1, m,     {:f, 32}, fn x -> 2 end)

prev = System.monotonic_time()

gpu_a = PolyHok.new_gnx(a)
gpu_x = PolyHok.new_gnx(x)

gpu_y = PGemv.gevm(gpu_a, gpu_x, m, n)

y = PolyHok.get_gnx(gpu_y)

next = System.monotonic_time()

if finput_print_result do
  IO.inspect(a)
  IO.inspect(x)
  IO.inspect(y)
end

IO.puts "time taken:\t#{System.convert_time_unit(next-prev,:native,:millisecond)}ms"
