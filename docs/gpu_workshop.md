# GPU computing with Julia: from memcopy to Cahn-Hilliard

**Julia4HPC workshop**, GPU part. Ludovic Räss

We take one equation, Cahn-Hilliard in 2D, and carry it from a plain CPU loop to a
GPU kernel running close to the memory bandwidth of the device, measuring at every
step. Everything is written with
[KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl), so the
same code runs on NVIDIA, AMD and Apple GPUs, and on the CPU.

Run the cells from top to bottom.

## Setup

Run this cell first, and again whenever you restart the kernel. It activates the
workshop environment and loads the packages we need.

````julia
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()

using KernelAbstractions
using CairoMakie
using Printf, Random, Statistics
````

Now pick **one** backend by uncommenting its line, and comment out the others.
The default is the CPU, which works everywhere. `FT` is the floating-point type
we compute in: Apple GPUs do not support `Float64`.

````julia
backend = CPU();                          FT = Float64  # CPU, no GPU needed
# using CUDA;   backend = CUDABackend();  FT = Float64  # NVIDIA (Arctic: ar_mig, ar_a100, ar_h200)
# using AMDGPU; backend = ROCBackend();   FT = Float64  # AMD    (Arctic: ar_mi210)
# using Metal;  backend = MetalBackend(); FT = Float32  # Apple laptop

println("backend = ", nameof(typeof(backend)), ",  FT = ", FT)
````

## 1. Why GPUs

## 2. The problem: Cahn-Hilliard in 2D

## 3. What limits performance

## 4. Memory bound: memcopy is the ceiling

## 5. Measuring memcopy without KernelAbstractions

## 6. A first kernel: memcopy with KernelAbstractions

````julia
@kernel inbounds = true function memcopy_ka!(A, B)
    ix, iy = @index(Global, NTuple)
    A[ix, iy] = B[ix, iy]
end

n = 256
A = KernelAbstractions.zeros(backend, FT, n, n)
B = KernelAbstractions.allocate(backend, FT, n, n)
copyto!(B, rand(FT, n, n))

k_memcopy! = memcopy_ka!(backend, 256, (n, n))
k_memcopy!(A, B)
KernelAbstractions.synchronize(backend)
@assert Array(A) == Array(B)
````

## 7. Cahn-Hilliard: discretisation and CPU reference

## 8. Cahn-Hilliard with array programming

## 9. Performance of array programming

## 10. Kernel programming with neighbours

## 11. Performance of kernel programming

## Outlook

