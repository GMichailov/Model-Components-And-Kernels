Swizzling is rearranging the order of data elements to improve memory access efficiency since GPU perf is dominated by memory access patterns.
All warps issue memory requests together, so best perf comes from coalesced accesses (want threads to access adjacent chunks of memory in order to isssue a single large memory transaction rather than many small ones).
Swizzling remaps indices so threads access memory ina hardware-friendly order.
The most common swizzle is the XOR swizzle which does: new_col = col^(row & mask) which slightly scrambles adresses so threads land in different memory banks and avoids shared memory bank conflicts.
SMEM has 32 banks where address mapping is bank = address % 32 and when two threads access the same bank, that causes serialization.

Thread swizzling changes which thread processes which element (used in warp shuffles and load balancing).
Memory swizzling changes layout of data in memory.

Swizzling is just a few integer ops like XOR, bit shifts, and bit masks which as 1-cycle ops on GPUs, so not expensive to do at all where hitting SMEM is 20 cycles comparatively.
Additionally, if swizzling avoids a bank conflict or an extra memory transaction, it saves hundreds of cycles.

In the context of ML, swizzling is just a deterministric index transform applied when writing and reading SMEM tiles.
When memory is permuted, threads know which element is which by never losing the mapping. 

The implementation is super simple where the swizzle is just part of the address calculation for shared memory.
In the example where a thread handles a singular row and col, it applies the swizzled col transformation, then accesses shared memory using row * stride + swizzled col.

## From Lei Mao Technical Blog

Swizzling will rearrange the mapping of the SMEM index to avoid bank conflicts and matrix transpose is a perfect example that can have an SMEM bank conflict if not using swizzling.

The goal of swizzling is to line it up so that there are no bank intersections horizontally and vertically.

### Formula

Given T array[][NX] on SMEM, SWIZZLE_SIZE = NX * sizeof(T) where the allowed vals of SWIZZLE_SIZE are powers of 2 >= 32 (32, 64, 128 etc). Given index [y][x] in T array[][NX], x_swz is computed as follows:
- Define Transaction Chunk -byte chunk size (typically larger than sizeof(T) ie 16 bytes if T is 2, 4, 8 bytes)
- Compute index of TC withing SWIZZLE_SIZE-byte segment.
  - i_chunk = (y * NX + x) * sizeof(T) / sizeof(TC) : Finds index of grouped chunk.
  - y_chunk = i / (SWIZZLE_SIZE / sizeof(TC)) : Finds where the chunk sits inside of the SWIZZLE_SIZE byte segment.
  - x_chunk = i % (SWIZZLE_SIZE / sizeof(TC))
- Compute the swizzled index of TC byte chunk using XOR:
  - x_chunk_swz = y_chunk ^ x_chunk: XOR mixes row and column indices to spread access across SMEM banks. XOR breaks the conflicts.
- Compute swizzled index
  - x_swz = x_chunk_swz * sizeof(TC) / sizeof(T) % NX + x % (sizeof(TC) / sizeof(T)) : converts swizzled chunk index back into element index.
 
### Properties
- Index before and afrer swizzling must be 1:1 mapped.
- NX must be power of 2.
- Given any x and any {y:y+31}, the number of unique swizzled index x_swz should be maximized.



