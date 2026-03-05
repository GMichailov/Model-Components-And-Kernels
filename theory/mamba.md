# Mamba-2

## Paper Summary

https://arxiv.org/pdf/2405.21060

State space model architecture designed to replace linear time due to replacing the quadratic compute and memory requirements of attention are quadratic and Mamba is linear.

Change in hidden state vector dh(t)/dt = state transition matrix A * hidden state vector h(t-1) + input projection matrix B * input signal token embedding.
Output representation y(t) = Readout matrix C(t) * hidden state matrix h(t).

State size is typically a latent (less than d_model) denoted by d_state.
A_not should belong to R^{d_state x d_state}, however as an efficiency trick, A is actually diag(A_not), so A is actually only belongs to R^{d_state}.
This lowers the parameter count from 16k (assuming d_state=128) to 128 and compute is a simple elementwise mul rather than a heavy GEMM.
B belongs to R^{d_state x d_model}.
C belongs to R^{d_model x d_state}.

## Kernel Trick

h1 = b1*x1, h2 = a2*h1 + b2*x2, h3 = a3*h2 + b3*x3 etc.

These can be rewritten as:
h1 = b1x1
h2 = a2b1x1 + b2x2
h3 = a3a2b1x1 + a3b2x2 + b3x3

This can be written as a matmul with matrix K such that H = K(B elementwise mul X) where:
K is a lower triangular matrix:
[1     , 0   ,  0, 0, 0, ... , L]
[a2    , 1   ,  0, 0, 0, ... , L]
[a3a2  , a3  , 1 , 0, 0, ... , L]
[a4a3a2, a4a3, a4, 1, 0, ... , L]
and so on (an L x L matrix) which is problematic. However, this matrix is just a cumulative product of a's, so can use block matrix multiplies.

Instead of doing a sequential scan, the paper says to split the text into blocks (ie 128), so if L=4096, there will be 32 blocks.

For each chunk
