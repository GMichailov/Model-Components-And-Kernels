# Cuda Streams

There are three steps to kernel execution:
- Memory copy from host to device.
- kernel execution.
- Memory copy from device to host.
Doing these in a serial manner is not the most efficient way to do this and that is why cuda streams exist.
By using a concurrent model with async memory transfers and kernel execution, they can overlap in time, so divide and process data in smaller chunks.
The H2D engine copies the first chunk of data from host onto the GPU and as soon as that copy finishes, the GPU launches a kernel to process that chunk.
However, while the kernel is active, the H2D copy engine is free again, so immediately start copying the second chunk onto GPU.
Same logic applies the D2H transfer engine. To ensure each chunk's ops happen in the correct order while still overlapping different chunks, cuda uses streams.

## Default Stream

### Legacy Version (Unlikely in Deep Learning Systems)

Work in the non-default streams remains async, however, work in the default stream waits for all other streams and all other streams wait for the default stream.
It basically acts like a global barrier stream. It forces all launches to work in order (relative to is/isn't default stream).

Kernels launched without specifying stream are automatically delegated to default. 
Supposed kernels A, B, C, and D are launched in this order where all run on non-default streams except C. This means that A and B can run simultaneously, C starts upon the completion of both,
and D only starts upon completion of C.

### Modern Version (Per-Thread Default Stream)

The default stream behaves like a normal stream, doesn't synchronize with other streams, and work runs concurrently across streams.
To enable this use: #define CUDA_API_PER_THREAD_DEFAULT_STREAM or --default-stream per-thread when compiling.

## Non-Default Stream
cudaStream_t stream; cudaStreamCreate(&stream);

## Number of Streams

All of the examples always show the classic 3 stream scenario where all 3 steps have an equal time cost, but this is almost never the true case.
Generally speaking for deep learning, compute is usually much heavier than copies.

Since the compute happens at the SM level, and we are assuming compute is Nx as heavy copying, the most optimal stream count is typically ceil(N) streams so that copying is always active and compute is always running.
The caveat to this is if compute is capped out like with large models, having multiple streams doesn't really deliver the advantage of having extra gpu cores idling. 

For example, something like FlashAttention is very compute heavy, so all SMs are already being utilized.
