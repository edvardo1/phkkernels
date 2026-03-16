import torch

def bench(m, n):
    A = torch.full([m, n], 2, device="cpu", dtype=torch.float32)
    x = torch.full([n], 2, device="cpu", dtype=torch.float32)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()
    start.record()

    A_gpu = A.to("cuda")
    x_gpu = x.to("cuda")

    y_gpu = torch.mv(A_gpu, x_gpu)
    y = y_gpu.to("cpu")

    end.record()
    torch.cuda.synchronize()

    ms = start.elapsed_time(end)

    if finput_print_result:
        print(A)
        print(x)
        print(y)

    return ms


t = bench(finput_m, finput_n)
print(f"torch: {t:.4f}ms")
