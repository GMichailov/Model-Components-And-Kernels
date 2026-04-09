# Tensor Storage

## Row/Column - Major

Row major means the matrix is stored contiguously from left to right, top to bottom, row by row.
Column major means the matrix is stored contiguously from top to bottom, left to right, column by column.

Even though the data is the same, the matmul operations are actually different depending on layout.

Given matrix A is (M, K) and matrix B is (K, N), C = AB. Each element in C is an accumulated sum of one row of size K in A and one column matrix B.
As you can probably guess from that last sentence, the fastest combination of row/col major A and row/col major B is row-major A and col-major B.
This has to do with the reading of data using vectorized-access which is only possible through contiguous loads.
