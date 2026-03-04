# Model-Components-And-Kernels

This is a public repo with pre-built widely used pytorch model components with efficient kernel implementations. I would like to add as much support as possible for different things, so this is an ongoing project and any contributions are welcome.

The plan is to write kernels in CUDA, Metal, ROCm and primarily focus on widely used components in modern models. I will target those used mainly in LLMs at the moment.

Planned order list:
- Mamba2
- Gated Delta Networks
- FlashAttention
- MoE
- Fused RmsNorm

Techniques:
- Async buff loading into Smem
- Swizzling
- Using CUB
