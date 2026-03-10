When writing a kernel that uses type T and you want to use SMEM, that means that smem needs to be of type T (extern __shared__ T smem_data[];), but this is not allowed.
One solution is to just declare extern __shared__ char smem[]; T* smem_data = reinterpret_cast<T*>(smem);
