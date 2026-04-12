# Cheat Sheet for cuda commands

## Allocating and Freeing Memory Memory

### Synchronous
- float* x; cudaMallocManaged(&x, N*sizeof(float)); : Allocates unified memory that can be accessed from bothCPU and GPU. Generally avoided in deep learning since it has performance unpredictability due to page migrations, unexpected transfers.
- cudaMalloc(&device_x, size); cudaMemcpy(device_x, host_x, size, cudaMemcpyHostToDevice); : Allocates explicit GPU memory and transfers this data from the host to device.
- cudaFree(device_x): Frees memory from device. This blocks CPU until completed.

### Asynchronous

Given cudaStream_t stream; cudaStreamCreate(&stream);
- cudaMallocAsync(&device_x, size, stream); cudaMemcpyAsync(device_x, host_x, size, cudaMemcpyHostToDevice, stream): Same as above, but non CPU blocking.
- cudaFreeAsync(device_x, stream);

#### Important Notes

- For true async behavior, host memory must be pinned using cudaMallocHost(&host_x, size); bc if memory is not pinned, cuda may fall back to sync copies.
- CUDA memory pools allow configuration of fragmentation behavior and reserved memory, so you can tell cuda things like keeping freed memory instead of returning it to the OS (massive save on allocation cost).
  - cudaMemPool_t pool; cudaDeviceGetDefaultMemPool(&pool, device);
  - size_t thresh = UINT64_MAX; cudaMemPoolSetAttribute(pool, cudaMemPoolAttributeReleaseThreshold, &thresh); ReleaseThreshold is the integer number of bytes of memory allowed to be cached in the memory pool.
 
## Synchronizing Threads

- cudaDeviceSynchronize(): Blocks CPU until GPU finishes all previously launched work.
