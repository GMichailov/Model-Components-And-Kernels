import math
import statistics
import torch
import torch.nn.functional as F
from ComponentsAndKernels.Kernels import fused_norms

# =============================================================================
# CONFIG
# =============================================================================

device = torch.device("cuda")
dtype = torch.bfloat16

seq_len = 5000
hidden_dim = 5120
epsilon = 1e-5

vec_sizes = [1, 2, 4, 8]
num_threads_options = [32, 64, 96, 128, 256, 512, 640, 1024]

num_warmup = 20
num_iters = 100          # timed iterations per repeat
num_repeats = 5          # number of timing repeats
x_pool_size = 8          # 8 * 3000 * 5120 * 2 bytes ~= 245.76 MB of inputs
do_l2_flush_between_repeats = True

# bf16 tolerances
atol = 2e-2
rtol = 2e-2

torch.set_grad_enabled(False)

# =============================================================================
# HELPERS
# =============================================================================

def bytes_per_tensor(shape, dtype):
    return math.prod(shape) * torch.tensor([], dtype=dtype).element_size()

def logical_bytes_per_call(x_shape, x_dtype, gamma_shape, gamma_dtype, y_dtype):
    """
    Logical minimum bytes moved per RMSNorm call:
      - read x
      - read gamma
      - write y

    This is best interpreted as EFFECTIVE BANDWIDTH, not true physical HBM traffic.
    """
    x_bytes = bytes_per_tensor(x_shape, x_dtype)
    gamma_bytes = bytes_per_tensor(gamma_shape, gamma_dtype)
    y_bytes = bytes_per_tensor(x_shape, y_dtype)
    return x_bytes + gamma_bytes + y_bytes

def max_abs_and_rel_err(y, ref):
    diff = (y.float() - ref.float()).abs()
    max_abs = diff.max().item()
    denom = ref.float().abs().clamp_min(1e-12)
    max_rel = (diff / denom).max().item()
    return max_abs, max_rel

@torch.inference_mode()
def baseline_rmsnorm_eager(x, gamma, eps):
    # Functional baseline: no module mutation, no Parameter cloning/swapping.
    return F.rms_norm(x, (x.shape[-1],), gamma, eps)

# torch.compile baseline
try:
    baseline_rmsnorm_compiled = torch.compile(baseline_rmsnorm_eager, dynamic=False)
except Exception:
    baseline_rmsnorm_compiled = baseline_rmsnorm_eager

# =============================================================================
# DATA POOLS
# =============================================================================

# A pool of inputs much larger than L2 to reduce cache-hit effects from reusing
# the exact same x every iteration.
x_pool = [
    torch.randn(seq_len, hidden_dim, device=device, dtype=dtype)
    for _ in range(x_pool_size)
]

# Gamma is tiny (~10 KB), so its cache behavior is negligible compared to x/y.
gamma_pool = [
    torch.randn(hidden_dim, device=device, dtype=dtype)
    for _ in range(x_pool_size)
]

# Optional L2 eviction buffer. This is outside the timed region.
# 128 MB is larger than L2 on current NVIDIA parts.
l2_flush_buf = torch.empty((128 * 1024 * 1024) // 4, device=device, dtype=torch.float32)

@torch.inference_mode()
def flush_l2():
    # Read+write a large buffer to evict prior data from L2.
    l2_flush_buf.add_(1.0)

# =============================================================================
# CORRECTNESS REFERENCE
# =============================================================================

# Use fp32 reference, then cast down to bf16 for comparison to your bf16 output.
with torch.inference_mode():
    x_ref = x_pool[0].float()
    gamma_ref = gamma_pool[0].float()
    ref_out = F.rms_norm(x_ref, (hidden_dim,), gamma_ref, epsilon).to(dtype)

# =============================================================================
# BENCH CORE
# =============================================================================

logical_bytes = logical_bytes_per_call(
    x_shape=(seq_len, hidden_dim),
    x_dtype=dtype,
    gamma_shape=(hidden_dim,),
    gamma_dtype=dtype,
    y_dtype=dtype,
)

@torch.inference_mode()
def benchmark_op(name, op):
    """
    Benchmarks end-to-end op latency.
    Note: this still includes output allocation cost if the op allocates internally.
    For pure kernel-only timing, your extension should expose an out= variant.
    """
    # Correctness check
    y = op(x_pool[0], gamma_pool[0], epsilon)
    torch.cuda.synchronize()

    max_abs, max_rel = max_abs_and_rel_err(y, ref_out)
    is_close = torch.allclose(y, ref_out, atol=atol, rtol=rtol)

    # Warmup
    for i in range(num_warmup):
        idx = i % x_pool_size
        _ = op(x_pool[idx], gamma_pool[idx], epsilon)
    torch.cuda.synchronize()

    per_repeat_ms = []

    for rep in range(num_repeats):
        if do_l2_flush_between_repeats:
            flush_l2()
            torch.cuda.synchronize()

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        torch.cuda.synchronize()
        start_event.record()

        for i in range(num_iters):
            idx = (rep * num_iters + i) % x_pool_size
            _ = op(x_pool[idx], gamma_pool[idx], epsilon)

        end_event.record()
        end_event.synchronize()

        elapsed_ms = start_event.elapsed_time(end_event)
        per_call_ms = elapsed_ms / num_iters
        per_repeat_ms.append(per_call_ms)

    median_ms = statistics.median(per_repeat_ms)
    mean_ms = statistics.mean(per_repeat_ms)
    min_ms = min(per_repeat_ms)
    max_ms = max(per_repeat_ms)
    stdev_ms = statistics.pstdev(per_repeat_ms) if len(per_repeat_ms) > 1 else 0.0

    effective_bw_gb_s = (logical_bytes / (median_ms / 1000.0)) / 1e9

    return {
        "name": name,
        "median_ms": median_ms,
        "mean_ms": mean_ms,
        "min_ms": min_ms,
        "max_ms": max_ms,
        "stdev_ms": stdev_ms,
        "effective_bw_gb_s": effective_bw_gb_s,
        "max_abs_err": max_abs,
        "max_rel_err": max_rel,
        "correct": is_close,
    }

# =============================================================================
# RUN BASELINES
# =============================================================================

print(
    f"\nRMSNorm benchmark"
    f"\n  dtype={dtype}"
    f"\n  shape=({seq_len}, {hidden_dim})"
    f"\n  warmup={num_warmup}"
    f"\n  iters/repeat={num_iters}"
    f"\n  repeats={num_repeats}"
    f"\n  x_pool_size={x_pool_size}"
    f"\n  logical bytes/call={logical_bytes / 1e6:.3f} MB"
    f"\n  L2 flush between repeats={do_l2_flush_between_repeats}"
)

results = []

eager_result = benchmark_op("PyTorch eager F.rms_norm", baseline_rmsnorm_eager)
results.append(eager_result)

compiled_result = benchmark_op("PyTorch compiled F.rms_norm", baseline_rmsnorm_compiled)
results.append(compiled_result)

baseline_median_ms = compiled_result["median_ms"]

print("\nBASELINES")
print("-" * 120)
print(
    f"{'Name':<34} {'Median ms':>10} {'Mean ms':>10} {'Min ms':>10} {'Max ms':>10} "
    f"{'Std ms':>10} {'Eff BW GB/s':>13} {'Correct':>9} {'MaxAbsErr':>12} {'MaxRelErr':>12}"
)
for r in results:
    print(
        f"{r['name']:<34} "
        f"{r['median_ms']:>10.4f} "
        f"{r['mean_ms']:>10.4f} "
        f"{r['min_ms']:>10.4f} "
        f"{r['max_ms']:>10.4f} "
        f"{r['stdev_ms']:>10.4f} "
        f"{r['effective_bw_gb_s']:>13.2f} "
        f"{str(r['correct']):>9} "
        f"{r['max_abs_err']:>12.6f} "
        f"{r['max_rel_err']:>12.6f}"
    )

# =============================================================================
# RUN CUSTOM KERNELS
# =============================================================================

custom_results = []

for vec in vec_sizes:
    func_name = f"rmsnorm_forward_bf16_bf16_fp32_v{vec}_dim5120"
    func = getattr(fused_norms, func_name)

    for threads in num_threads_options:
        def op(x, gamma, eps, _func=func, _threads=threads):
            return _func(x, gamma, eps, _threads)

        name = f"custom v{vec} threads={threads}"

        try:
            result = benchmark_op(name, op)
            result["vec"] = vec
            result["threads"] = threads
            result["speedup_vs_compiled"] = baseline_median_ms / result["median_ms"]
            custom_results.append(result)
        except Exception as e:
            print(f"Skipping {name}: {e}")

# Keep only correct results for ranking
rankable = [r for r in custom_results if r["correct"]]
rankable.sort(key=lambda r: r["median_ms"])
top10 = rankable[:10]

print("\nTOP 10 FASTEST CORRECT CUSTOM CONFIGURATIONS")
print("-" * 140)
print(
    f"{'Rank':<6} {'Vec':<6} {'Threads':<10} {'Median ms':>10} {'Mean ms':>10} "
    f"{'Min ms':>10} {'Max ms':>10} {'Std ms':>10} {'Eff BW GB/s':>13} "
    f"{'vs compiled':>12} {'MaxAbsErr':>12} {'MaxRelErr':>12}"
)

for rank, r in enumerate(top10, 1):
    print(
        f"{rank:<6} "
        f"{r['vec']:<6} "
        f"{r['threads']:<10} "
        f"{r['median_ms']:>10.4f} "
        f"{r['mean_ms']:>10.4f} "
        f"{r['min_ms']:>10.4f} "
        f"{r['max_ms']:>10.4f} "
        f"{r['stdev_ms']:>10.4f} "
        f"{r['effective_bw_gb_s']:>13.2f} "
        f"{r['speedup_vs_compiled']:>12.2f}x "
        f"{r['max_abs_err']:>12.6f} "
        f"{r['max_rel_err']:>12.6f}"
    )

# Optional: show incorrect configs too
bad = [r for r in custom_results if not r["correct"]]
if bad:
    print("\nINCORRECT / FAILED CORRECTNESS CHECK")
    print("-" * 120)
    for r in bad:
        print(
            f"{r['name']:<28} "
            f"median_ms={r['median_ms']:.4f}, "
            f"max_abs_err={r['max_abs_err']:.6f}, "
            f"max_rel_err={r['max_rel_err']:.6f}"
        )