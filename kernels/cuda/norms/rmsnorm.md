# RMSNorm

This is just for me to implement as practice since it is one of the easiest to do.

## Forward Formula

RMSNorm(x) = gamma (elementwise_mul) * x / (RMS(x) + epsilon)
- gamma is 1 by d_model (or whatever the last dim of x is [important to mention due to it being applied to latents, etc.])
- epsilon is const scalar (typically 1e-6 or 1e-8 to avoid root(0))
- RMS(x): Root mean square of x (1/d_x * SUM(x_i^2 for i in range(0, d_x)))

## Backward Formula



