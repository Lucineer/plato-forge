# PLATO Forge — GPU Kernel Room

Submit CUDA/PTX kernels → benchmark on available GPU → track results.

## Actions
- `submit` — kernel source (cuda/ptx), name, description
- `benchmark` — compile and profile a submitted kernel

## Safety
- Blocks dangerous patterns (system, exec, file I/O)
- Requires __global__ (CUDA) or .entry (PTX)
- Sandboxed compilation via ptxas/nvcc to /dev/null

## Fleet
- **JC1**: benchmark on Jetson sm_87
- **Forgemaster**: benchmark on RTX 4050 sm_89
