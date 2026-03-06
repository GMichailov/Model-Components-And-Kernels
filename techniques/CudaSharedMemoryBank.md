## Memory Bank

SMEM is divided into equally sized memory banks that can be accessed simultaneously for concurrent access.
If multiple addresses of a memory request map to the same memory bank, the access are serialized (multiple sequential memory read ops).
This means bank conflicts can multiply the amount of time to read by several times.
(The only exception is when threads in the SAME warp access the same memory bank. This results in a broadcast from different banks coalesced into a single multicast to all threads in the warp).

Just about all (enterprise and consumer) GPUs  have a bank bandwidth of 32 bits every clock cycle (so  4 bytes * 32 banks = 128 bytes throughput per cycle).
Memory addresses map to banks in a simple pattern where bank = (address / 4 bytes since everything functions in bytes) mod 32.
A warp is 32 threads and they execute the same instruction simultaneously, so a read like read_smem[base_index + thread_id] results in 32 simultaneous memory accesses.
The best case scenario is one thread per bank (like above) where each accesses its own distinct bank, so all reads occur in parallel in one cycle (conflict-free access).
The worst case conflict is all 32 threads read the same bank, but in this special case, the hardware does a broadcast for it to still be one cycle.
