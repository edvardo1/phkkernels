import torch

def bench(m, n):
    x = torch.full([m], 2, device="cpu", dtype = torch.float32)
    A = torch.full([m, n], 2, device="cpu", dtype = torch.float32)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()
    start.record()

    x_gpu = x.to("cuda")
    A_gpu = A.to("cuda")

    c_gpu = torch.matmul(x_gpu, A_gpu)
    c = c_gpu.to("cpu")

    end.record()
    torch.cuda.synchronize()

    ms = start.elapsed_time(end)

    if finput_print_result:
        print(x)
        print(A)
        print(c)

    return ms


t = bench(finput_m, finput_n)
print(f"torch: {t:.4f}ms")
