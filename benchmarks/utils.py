from dataclasses import dataclass, field
from enum import Enum
from typing import List
import torch
import math
import subprocess

class ThreadOptions(Enum):
    T16 = 16
    T32 = 32
    T64 = 64
    T96 = 96
    T128 = 128
    T256 = 256
    T512 = 512
    T768 = 768
    T1024 = 1024

@dataclass
class BenchmarkConfig:
    device: str = "cuda"
    dtypes: List[torch.dtype] = field(default_factory=lambda: [
        torch.bfloat16,
        torch.float16,
        torch.float32,
    ])
    batch_size: List[int] = field(default_factory=lambda: [1, 2, 4, 8, 16, 32])
    seq_len: List[int] = field(default_factory=lambda: [128, 256, 512, 1024, 2048, 4096])
    d_model: List[int] = field(default_factory=lambda: [
        128, 256, 384, 512, 768, 1024, 1280, 1536,
        2048, 2560, 3072, 4096, 5120, 6144, 8192
    ])
    threads_per_block: List[int] = field(default_factory=lambda: [
        32, 64, 128, 256, 512, 768, 1024
    ])
    vectorized_load_sizes: List[int] = field(default_factory=lambda: [1, 2, 4, 8])
    warmup_iters: int = 20
    benchmark_iters: int = 100
    timing_repeats: int = 5
    atol: float = 1e-3
    rtol: float = 1e-3

def run_compile():
    result = subprocess.run(
        ["python3", "setup.py", "build_ext", "--inplace"],
        cwd="/home/george/repos/Model-Components-And-Kernels",
        check=True
    )
    return result.returncode

def bytes_per_tensor(shape, dtype):
    return math.prod(shape) * torch.tensor([], dtype=dtype).element_size()

def max_abs_and_rel_err(y, ref):
    diff = (y.float() - ref.float()).abs()
    max_abs = diff.max().item()
    denom = ref.float().abs().clamp_min(1e-12)
    max_rel = (diff / denom).max().item()
    return max_abs, max_rel

@torch.inference_mode()
def flush_l2():
    # Read+write a large buffer to evict prior data from L2. (48 MB is bigger than A100 already)
    l2_flush_buf = torch.empty((48 * 1024 * 1024) // 4, device="cuda", dtype=torch.float32)
    l2_flush_buf.add_(1.0)