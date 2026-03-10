Grids wrap a bunch of blocks, a block wraps a bunch of threads, and a thread handles a small amount of memory-based work.

An interesting point when choosing dims to consider is parallelism. 
As an example, take RMSNorm where dependents are across d_model per token and the [Batch, Seq, d_model] array
is completely independent across the first two dims. 

Assuming the hypothetical situation where an equal number of threads are used, it is actually better to launch more blocks because
each block runs on a singular Streaming Multiprocessor. Therefore, it is important to delegate work across blocks, so they can work in parallel.
The compiler handles scheduling to hide latencies.

The true optimal dimensions are usually found by tuning on the exact dims running (ie 1 block per row when d_model is 16 is inefficient and we want them to handle more)
while with a SOTA 1T param model like KimiK2.5, we might even delegate 1 block per row with the max number of threads.

Block looping is most usually correct in persistent kernels like flash attention.
