import torch

def bench(m, n, k):
    A = torch.full([m, k], 2, device="cpu", dtype=torch.float32)
    B = torch.full([k, n], 2, device="cpu", dtype=torch.float32)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()
    start.record()

    A_gpu = A.to("cuda")
    B_gpu = B.to("cuda")

    C_gpu = torch.mm(A_gpu, B_gpu)
    C = C_gpu.to("cpu")

    end.record()
    torch.cuda.synchronize()

    ms = start.elapsed_time(end)

    if finput_print_result:
        print(A)
        print(B)
        print(C)

    return ms


t = bench(finput_m, finput_n, finput_k)
print(f"torch: {t:.4f}ms")
