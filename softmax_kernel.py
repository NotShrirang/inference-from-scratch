import time

import torch
import triton
from triton import language as tl


@triton.jit
def softmax_kernel(X_ptr, out_ptr, n_cols, BLOCK_SIZE: tl.constexpr = None):
    pid = tl.program_id(0)
    
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols
    offsets = pid * n_cols + cols

    # Load input values
    x = tl.load(X_ptr + offsets, mask=mask, other=-float("inf"))

    # Compute max for numerical stability
    max_val = tl.max(x, axis=0)
    x = x - max_val

    # Compute exponentials
    exp_x = tl.exp(x)

    # Compute sum of exponentials
    sum_exp_x = tl.sum(exp_x, axis=0)

    # Compute softmax output
    softmax_out = exp_x / sum_exp_x

    # Store output values
    tl.store(out_ptr + offsets, softmax_out, mask=mask)



def softmax(X, BLOCK_SIZE=None):
    n_rows, n_cols = X.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols) if BLOCK_SIZE is None else BLOCK_SIZE
    out = torch.empty_like(X)
    grid = (n_rows,)
    softmax_kernel[grid](X, out, n_cols, BLOCK_SIZE)

    return out


def benchmark_softmax(sizes, times, device='cuda'):
    results = []
    for size in sizes:
        X = torch.randn(size, device=device)
        # Warm-up
        softmax(X)
        torch.cuda.synchronize()

        # Benchmark Triton softmax
        start_time = time.time()
        for _ in range(times):
            triton_out = softmax(X)
        torch.cuda.synchronize()
        triton_time = (time.time() - start_time) / times

        # Benchmark PyTorch softmax
        start_time = time.time()
        for _ in range(times):
            torch_out = torch.nn.functional.softmax(X, dim=1)
        torch.cuda.synchronize()
        torch_time = (time.time() - start_time) / times

        allclose = torch.allclose(triton_out, torch_out, atol=1e-6)

        results.append({
            'size': size,
            'triton_time': triton_time,
            'torch_time': torch_time,
            'speedup': torch_time / triton_time,
            'allclose': allclose
        })
    return results


if __name__ == "__main__":
    sizes = [(1024, 512), (2048, 1024), (4096, 2048), (8192, 4096)]
    times = 100
    results = benchmark_softmax(sizes, times)

    for result in results:
        print(f"Size: {result['size']}, Triton Time: {result['triton_time']:.6f}s, "
              f"PyTorch Time: {result['torch_time']:.6f}s, Speedup: {result['speedup']:.2f}x, "
              f"All Close: {result['allclose']}")

    X = torch.randn(1024, 512, device='cuda')

    import triton.testing
    t_triton = triton.testing.do_bench(lambda: softmax(X))      # ms
    t_torch  = triton.testing.do_bench(lambda: torch.softmax(X, dim=1))

    print(f"Triton softmax time: {t_triton:.6f} ms")
    print(f"PyTorch softmax time: {t_torch:.6f} ms")
    print(f"Speedup: {t_torch / t_triton:.2f}x")
