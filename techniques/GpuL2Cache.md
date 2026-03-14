# L2 Cache

Starting with CUDA 11 and compute >= 8.0, you can reserve part of L2 as a set-aside for persisting acesses and then mark
a specific global-memory window so accesses there are more likely to be retained in that reserved portion. Thist can reduce repeated DRAM traffic for small, but frequrently reused data structs.

## Persistent L2

Normally, the L2 is a hardware-managed cache where loads come in and the GPU decides what to keep and what to evict.
Persistent L2 tells the hardware that a region is special and to try to avoid evicting things in it, but does NOT guarrantee that it will not
be evicted. If the data is too large, too random, or not reused enough before eviction pressure mounts, it will still likely get evicted.

Reserving it looks something like the following:
cudaDeviceProp prop; cudaGetDeviceProperties(&prop, device_id);
size_t size = min(int(prop.l2CacheSize * 0.75), prop.persistingL2CacheMaxSize);
cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, size);

This CAN help, however, it is a double edged sword. By taking this L2 cache, everything else has to effectively use a smaller L2 which can actually hurt performance rather than help.

## Access Policy Window

This is where contiguous global-memory region is specified and told how to treat access in it.
Things like base_ptr, num_bytes, hitRatio, hitProp, missProp can be set. This is the data that is to be put in the persistent cache.
This is called hot data. If the hot data is bigger than the reserved L2 space and hitRatio=1.0 (or close to 1.0) and CUDA tries to treat all of it as persistent which causes thrashing 
(new cache lines come in and evict older labeled-as-persistent data gets evicted.

- hitProp is what cache policy to use for the access that land in the "favored" subset of the window.
- missProp is what cache policy to use for the rest of the accesses in that same window.
- hitRatio is what fraction of the accesses in the window should get hitProp instead of missProp. Again, this is a hit and is stil ultimately up to the hardware.

This is created in the following order:
cudaStream_t stream; cudaStreamCreate(&stream);
cudaStreamAttrValue stream_attribute;
stream_attribute.accessPolicyWindow.base_ptr=reinterpret_cast<void*>(device_global_mem_persistent_target);
stream_attribute.accessPolicyWindow.num_bytes = num_kilobytes * 1024;
stream_attribute.accessPolicyWindow.hitRatio = float[0.0-1.0];
stream_attribute.accessPolicyWindow.hitProp = cudaAccesPropertyPersisting;
stream_attribute.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &stream_attribute);

The general anti-thrash setup is the following:
hitProp = cudaAccessPropertyPersisting;
missProp = cudaAccessPropertyStreaming;
hitRatio = min(reserved_persistent_L2_bytes / window_bytes, 1.0);

Simply put, if the L2 cache reservation is bigger, just pull everything in and persist. If the opposite is true,
use persistent behavior for as much of the window as possible.

Also note to be careful about reserving L2 when using multiple streams.


