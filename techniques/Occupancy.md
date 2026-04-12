# Calculating Occupancy Examples

## RTX 4060

24 SMs, 65536 32-bit registers.
minimum register allocation unit per thread = 8 thread. Meaning even if a thread needs 13 registers, it will get the next largest multiple of 8.
Suppose 128-thread block and each thread uses 20 registers.

Actual registers per thread = ceil(20/8) * 8 = 24.
Registers per warp = 32 threads per warp * 24 registers = 768 registers.
Registers per block = 3072 registers.
Max number of active blocks per SM = int(65536 / 3072) = 16.

You can get the number of registers used per thread for each kernel using --ptxas-option=-v when compiling with nvvcc.
