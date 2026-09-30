# # GPU computing with Julia: from memcopy to Cahn-Hilliard
#
# **Julia4HPC workshop**, GPU part. Ludovic Räss
#
# We take one equation, Cahn-Hilliard in 2D, and carry it from a plain CPU loop to a
# GPU kernel running close to the memory bandwidth of the device, measuring at every
# step. Everything is written with
# [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl), so the
# same code runs on NVIDIA, AMD and Apple GPUs, and on the CPU.
#
# Run the cells from top to bottom.
#hint #
#hint # **Exercises:** cells containing `???` are blanks to fill in. Stuck? Open the
#hint # [solution notebook](../solutions/gpu_workshop_solution.ipynb) next to this one.

#src ---------------------------------------------------------------------------------
#src Author notes: lines starting with #src never reach the generated outputs.
#src
#src Time budget, 2 x 80 min:
#src   Slot 1: setup 20' | S1-4 12' | S5 13' | S6 18' | S7 17'
#src   Slot 2: recap 5'  | S8 15'   | S9 10' | S10 25' | S11 15' | outlook 5' + 5' buffer
#src
#src Exercise markers, see deploy/deploy.jl:
#src   <code>  #sol    the answer, only in the solution
#src   #hint <code>    the blank, only in the exercise
#src ---------------------------------------------------------------------------------

# ## Setup
#
# Run this cell first, and again whenever you restart the kernel. It activates the
# workshop environment and loads the packages we need.

#src TODO: align with the shared depot on Arctic (Manifest.toml, whether to instantiate here).
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()

using KernelAbstractions
using CairoMakie
using Printf, Random, Statistics

# Now pick **one** backend by uncommenting its line, and comment out the others.
# The default is the CPU, which works everywhere. `FT` is the floating-point type
# we compute in: Apple GPUs do not support `Float64`.

backend = CPU();                          FT = Float64  # CPU, no GPU needed
## using CUDA;   backend = CUDABackend();  FT = Float64  # NVIDIA (Arctic: ar_mig, ar_a100, ar_h200)
## using AMDGPU; backend = ROCBackend();   FT = Float64  # AMD    (Arctic: ar_mi210)
## using Metal;  backend = MetalBackend(); FT = Float32  # Apple laptop

println("backend = ", nameof(typeof(backend)), ",  FT = ", FT)

# ## 1. Why GPUs
#src Talk only. PDE cost growth (CH explicit dt ~ dx^4), memory wall, CPU vs GPU bandwidth.

# ## 2. The problem: Cahn-Hilliard in 2D
#src Talk only. Equation, the C-bar regimes (gifs in assets/), invariants F and mean.

# ## 3. What limits performance
#src Talk only. Flops vs bytes, machine balance, cost of a 5-point stencil.

# ## 4. Memory bound: memcopy is the ceiling
#src Talk only. T_eff definition, counting arrays: memcopy 2, saxpy 3, CH 5.

# ## 5. Measuring memcopy without KernelAbstractions
#src Run only. copyto! and A .= B on the vendor array, size sweep, hand-written timing
#src loop (warm-up + synchronize). Output: T_peak, the reference for everything below.

# ## 6. A first kernel: memcopy with KernelAbstractions
#src Placeholder that exercises the #sol/#hint pipeline and CI. Rewrite with the content.

@kernel inbounds = true function memcopy_ka!(A, B)
    ix, iy = @index(Global, NTuple)  #sol
    #hint ix, iy = ???   # the 2D index of this work item
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

# ## 7. Cahn-Hilliard: discretisation and CPU reference
#src Grid units, 5-point Laplacian, ghost-node mirror, explicit dt limit, invariants.
#src Plain loop solver at 256^2; prints reference F and mean after N steps.
#src Blanks: mu and C updates, dt limit.

# ## 8. Cahn-Hilliard with array programming
#src KA.allocate + broadcasting, flux form (face arrays, zero boundary flux = mirror).
#src Blanks: flux and update broadcasts. Check against the S7 reference.

# ## 9. Performance of array programming
#src Bench each broadcast line against T_peak, blank: arrays moved per line.
#src Whole step T_eff with the minimal 5 arrays -> each line fast, the step is not.

# ## 10. Kernel programming with neighbours
#src What @index is: 1D launch + CartesianIndices tile (@index Group/Local demo),
#src static vs dynamic sizes, min/max boundaries, inbounds + @propagate_inbounds.
#src Metal-only Int32 shift/mask cell. Blanks: S7 loop body -> kernel body.

# ## 11. Performance of kernel programming
#src Bench each kernel, full step, n sweep, final T_eff plot of all variants vs T_peak.
#src Blanks: same pattern as S9.

# ## Outlook
#src Chmy.jl, ParallelStencil + ImplicitGlobalGrid (multi-GPU), Reactant.
#src Links: JuliaCon26-GPUs-for-HPC, pde-on-gpu course.
