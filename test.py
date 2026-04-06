import torch
from ComponentsAndKernels.Kernels import fused_norms

def rms_norm_reference_fp32(x, gamma, epsilon=1e-5):
    """Reference RMS Norm - computes in FP32, casts back to input dtype."""
    xf = x.float()
    gf = gamma.float()
    variance = xf.pow(2).mean(dim=-1, keepdim=True)
    return (xf * torch.rsqrt(variance + epsilon) * gf).to(x.dtype)

print("=" * 60)
print("RMS Norm Kernel Diagnostics (FP32 Reference)")
print("=" * 60)

# Test 1: Scalar loads/stores only (v1 variants)
print("\n[TEST 1] SCALAR LOADS/STORES (no vectorization)")
print("-" * 40)
for name, dtype, weight_dtype, kernel_fn in [
    ("fp16", torch.float16, torch.float16, fused_norms.rmsnorm_forward_fp16_fp16_fp32_v1),
    ("bf16", torch.bfloat16, torch.bfloat16, fused_norms.rmsnorm_forward_bf16_bf16_fp32_v1),
]:
    x = torch.randn(32, 128, dtype=dtype, device='cuda')
    gamma = torch.randn(128, dtype=weight_dtype, device='cuda')
    y_custom = kernel_fn(x, gamma, 1e-5, 256)
    y_ref = rms_norm_reference_fp32(x, gamma, 1e-5)
    max_diff = (y_custom - y_ref).abs().max().item()
    print(f"  {name}: max_diff = {max_diff:.6e}  {'PASS' if max_diff < 1e-3 else 'FAIL'}")

# Test 2: Vectorized loads/stores (v4 variants)
print("\n[TEST 2] VECTORIZED LOADS/STORES (v4 = 4 elements)")
print("-" * 40)
for name, dtype, weight_dtype, kernel_fn in [
    ("fp16_v4", torch.float16, torch.float16, fused_norms.rmsnorm_forward_fp16_fp16_fp32_v4),
    ("bf16_v4", torch.bfloat16, torch.bfloat16, fused_norms.rmsnorm_forward_bf16_bf16_fp32_v4),
]:
    x = torch.randn(32, 128, dtype=dtype, device='cuda')
    gamma = torch.randn(128, dtype=weight_dtype, device='cuda')
    y_custom = kernel_fn(x, gamma, 1e-5, 256)
    y_ref = rms_norm_reference_fp32(x, gamma, 1e-5)
    max_diff = (y_custom - y_ref).abs().max().item()
    print(f"  {name}: max_diff = {max_diff:.6e}  {'PASS' if max_diff < 1e-3 else 'FAIL'}")

# Test 3: 8-element vectorization (v8 variants - half/bf16 only)
print("\n[TEST 3] 8-ELEMENT VECTORIZATION (v8)")
print("-" * 40)
for name, dtype, weight_dtype, kernel_fn in [
    ("fp16_v8", torch.float16, torch.float16, fused_norms.rmsnorm_forward_fp16_fp16_fp32_v8),
    ("bf16_v8", torch.bfloat16, torch.bfloat16, fused_norms.rmsnorm_forward_bf16_bf16_fp32_v8),
]:
    x = torch.randn(32, 128, dtype=dtype, device='cuda')
    gamma = torch.randn(128, dtype=weight_dtype, device='cuda')
    y_custom = kernel_fn(x, gamma, 1e-5, 256)
    y_ref = rms_norm_reference_fp32(x, gamma, 1e-5)
    max_diff = (y_custom - y_ref).abs().max().item()
    print(f"  {name}: max_diff = {max_diff:.6e}  {'PASS' if max_diff < 1e-3 else 'FAIL'}")

# Test 4: gamma = 1 (isolates the packed store issue)
print("\n[TEST 4] GAMMA = 1 (checks reduction + store, not gamma path)")
print("-" * 40)
for name, dtype, kernel_fn in [
    ("fp16_v4", torch.float16, fused_norms.rmsnorm_forward_fp16_fp16_fp32_v4),
    ("bf16_v4", torch.bfloat16, fused_norms.rmsnorm_forward_bf16_bf16_fp32_v4),
]:
    x = torch.randn(32, 128, dtype=dtype, device='cuda')
    gamma = torch.ones(128, dtype=dtype, device='cuda')
    y_custom = kernel_fn(x, gamma, 1e-5, 256)
    y_ref = rms_norm_reference_fp32(x, gamma, 1e-5)
    max_diff = (y_custom - y_ref).abs().max().item()
    print(f"  {name}: max_diff = {max_diff:.6e}  {'PASS' if max_diff < 1e-3 else 'FAIL'}")

# Test 5: Random gamma, scalar stores
print("\n[TEST 5] RANDOM GAMMA + SCALAR STORES (v1 variants)")
print("-" * 40)
for name, dtype, weight_dtype, kernel_fn in [
    ("fp16_v1", torch.float16, torch.float16, fused_norms.rmsnorm_forward_fp16_fp16_fp32_v1),
    ("bf16_v1", torch.bfloat16, torch.bfloat16, fused_norms.rmsnorm_forward_bf16_bf16_fp32_v1),
]:
    x = torch.randn(32, 128, dtype=dtype, device='cuda')
    gamma = torch.randn(128, dtype=weight_dtype, device='cuda')
    y_custom = kernel_fn(x, gamma, 1e-5, 256)
    y_ref = rms_norm_reference_fp32(x, gamma, 1e-5)
    max_diff = (y_custom - y_ref).abs().max().item()
    print(f"  {name}: max_diff = {max_diff:.6e}  {'PASS' if max_diff < 1e-3 else 'FAIL'}")

print("\n" + "=" * 60)
print("DIAGNOSIS KEY:")
print("- If TEST 1 fails: vectorized loads are the problem")
print("- If TEST 4 passes but TEST 2 fails: packed store path is broken")
print("- If TEST 4 passes but TEST 5 fails: gamma read/unpack is broken")
print("=" * 60)
