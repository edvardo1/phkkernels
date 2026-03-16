require PolyHok

PolyHok.defmodule PGemv do
  defk gemvk(a, x, y, m, n) do
    warp = 32
    block_size = 256
    fullint = 0xffffffff
  
    row = blockIdx.x
    #a_row = a + row * n

    sum = 0.0
    col = threadIdx.x
    while col < n do
      #sum = sum + a_row[col] * x[col]
      sum = sum + a[row * n + col] * x[col]

      col = col + blockDim.x
    end
      
    off = warp / 2
    while (off > 0) do
        sum = sum + __shfl_down_sync(fullint, sum, off)
        off = off / 2
    end

    __shared__ buf[8]
    
    condition = (threadIdx.x / 32) * 32 == threadIdx.x
    if condition do
      buf[threadIdx.x / warp] = sum
    end
    
    __syncthreads()

    if (threadIdx.x < warp) do
      v = 0.0
      if threadIdx.x < blockDim.x / warp do
        v = buf[threadIdx.x]
      else
        v = 0.0
      end

      off = warp / 2
      while off > 0 do
        v = v + __shfl_down_sync(fullint, v, off);
        off = off / 2
      end
      if (threadIdx.x == 0) do
        y[row] = v
      end
    end
  end
  
  def gemv(a, x, m, n) do
    block_size = 256
    
    grid = m;
    block = block_size
    
    y = PolyHok.new_gnx(1, m, {:f, 32})
    PolyHok.spawn(&PGemv.gemvk/5,
                  {grid, 1, 1},
                  {block, 1, 1},
                  [a, x, y, m, n])
    y
  end
end

m = finput_m;
n = finput_n;

#mat = PolyHok.new_nx_from_function_arg(m, n, type: {:f, 32}, fn x -> x end)
mat = PolyHok.new_nx_from_function_arg(m, n, {:f, 32}, fn x -> x  end)
vec = PolyHok.new_nx_from_function_arg(1, n, {:f, 32}, fn x -> x end)

prev = System.monotonic_time()

gmat = PolyHok.new_gnx(mat)
gvec = PolyHok.new_gnx(vec) 

ovec_ = PGemv.gemv(gmat, gvec, m, n)
ovec = PolyHok.get_gnx(ovec_)

next = System.monotonic_time()

if finput_print_result do
  IO.inspect mat
  IO.inspect vec
  IO.inspect ovec
end

IO.puts "phk:\t#{System.convert_time_unit(next-prev,:native,:millisecond)}ms"
