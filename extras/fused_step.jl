# Cahn-Hilliard with a fused time step: one kernel per step, μ recomputed instead
# of stored. Reference solution for "Challenge: even fewer arrays", at the end of
# notebooks/gpu_workshop.ipynb.
#
# Run from the repository root:
#
#   julia --project=. extras/fused_step.jl
#
# Choose the backend below, as in the notebook.
using KernelAbstractions, Printf, Random, Statistics

backend = CPU();  FT = Float64                  # CPU: only to check that the code runs
# using CUDA;   backend = CUDABackend();  FT = Float64
# using AMDGPU; backend = ROCBackend();   FT = Float64
# using Metal;  backend = MetalBackend(); FT = Float32; include(joinpath(@__DIR__, "metal_index_fix.jl"))

# ---- the building blocks of the notebook ----

# Laplacian with mirror (no-flux) boundaries
Base.@propagate_inbounds function lap(A, ix, iy, nx, ny)
    a = A[ix, iy]
    return (A[max(ix-1, 1), iy] - 2a + A[min(ix+1, nx), iy]) +
           (A[ix, max(iy-1, 1)] - 2a + A[ix, min(iy+1, ny)])
end

# chemical potential μ = C³ - C - γ∇²C at one cell, computed from C
Base.@propagate_inbounds function mu(C, ix, iy, nx, ny, γ)
    c = C[ix, iy]
    return c^3 - c - γ * lap(C, ix, iy, nx, ny)
end

# the two kernels of the notebook (section 10): μ is stored between the passes
@kernel inbounds = true function potential_ka!(μ, C, γ)
    ix, iy = @index(Global, NTuple)
    nx, ny = size(C)
    μ[ix, iy] = mu(C, ix, iy, nx, ny, γ)
end

@kernel inbounds = true function concentration_ka!(C, μ, dtD)
    ix, iy = @index(Global, NTuple)
    nx, ny = size(C)
    C[ix, iy] += dtD * lap(μ, ix, iy, nx, ny)
end

# ---- fused, version 1: μ recomputed at the cell and its 4 neighbours ----
# Reads C, writes C2: 2 arrays per step instead of 5. C2 must be a second array,
# since the neighbours two cells away still need the old C.

@kernel inbounds = true function step_fused!(C2, C, γ, dtD)
    ix, iy = @index(Global, NTuple)
    nx, ny = size(C)
    μc   = mu(C, ix, iy, nx, ny, γ)
    lapμ = (mu(C, max(ix-1, 1), iy, nx, ny, γ) - 2μc + mu(C, min(ix+1, nx), iy, nx, ny, γ)) +
           (mu(C, ix, max(iy-1, 1), nx, ny, γ) - 2μc + mu(C, ix, min(iy+1, ny), nx, ny, γ))
    C2[ix, iy] = C[ix, iy] + dtD * lapμ
end

# ---- fused, version 2: plain offsets in the interior ----
# Two cells or more away from the boundary, no min/max is needed. With plain
# offsets, the compiler sees that many neighbour loads are the same and shares them.

Base.@propagate_inbounds function lap_interior(A, ix, iy)
    a = A[ix, iy]
    return (A[ix-1, iy] - 2a + A[ix+1, iy]) + (A[ix, iy-1] - 2a + A[ix, iy+1])
end

Base.@propagate_inbounds function mu_interior(C, ix, iy, γ)
    c = C[ix, iy]
    return c^3 - c - γ * lap_interior(C, ix, iy)
end

@kernel inbounds = true function step_fused_interior!(C2, C, γ, dtD)
    ix, iy = @index(Global, NTuple)
    nx, ny = size(C)
    if 2 < ix < nx - 1 && 2 < iy < ny - 1           # interior: plain offsets
        μc   = mu_interior(C, ix, iy, γ)
        lapμ = (mu_interior(C, ix-1, iy, γ) - 2μc + mu_interior(C, ix+1, iy, γ)) +
               (mu_interior(C, ix, iy-1, γ) - 2μc + mu_interior(C, ix, iy+1, γ))
    else                                            # near the boundary: min/max
        μc   = mu(C, ix, iy, nx, ny, γ)
        lapμ = (mu(C, max(ix-1, 1), iy, nx, ny, γ) - 2μc + mu(C, min(ix+1, nx), iy, nx, ny, γ)) +
               (mu(C, ix, max(iy-1, 1), nx, ny, γ) - 2μc + mu(C, ix, min(iy+1, ny), nx, ny, γ))
    end
    C2[ix, iy] = C[ix, iy] + dtD * lapμ
end

# ---- setup, as in the notebook ----

function initial_condition(FT, n; C̄=0, ampl=0.02, seed=1234)
    Random.seed!(seed)
    C = C̄ .+ FT(ampl) .* randn(FT, n, n)
    C .+= C̄ - mean(C)
    return C
end

function time_it(f, backend; nrep=20, ntrial=3)
    f()
    KernelAbstractions.synchronize(backend)
    t_best = Inf
    for trial in 1:ntrial
        t0 = time()
        for rep in 1:nrep
            f()
        end
        KernelAbstractions.synchronize(backend)
        t_best = min(t_best, (time() - t0) / nrep)
    end
    return t_best
end

D     = FT(1)
γ     = FT(4)^2 / 8
κmax  = FT(8)
dt    = 2 / (D * κmax * (γ * κmax + 2)) / 2
dtD   = dt * D

# ---- 1. same result as the two kernels ----

function run_two_kernels(backend, C0, γ, dtD, nt)
    n = size(C0, 1)
    C = KernelAbstractions.allocate(backend, eltype(C0), n, n); copyto!(C, C0)
    μ = KernelAbstractions.zeros(backend, eltype(C0), n, n)
    potential!     = potential_ka!(backend, 256, (n, n))
    concentration! = concentration_ka!(backend, 256, (n, n))
    for it in 1:nt
        potential!(μ, C, γ)
        concentration!(C, μ, dtD)
    end
    return Array(C)
end

function run_fused(kernel, backend, C0, γ, dtD, nt)
    n  = size(C0, 1)
    C  = KernelAbstractions.allocate(backend, eltype(C0), n, n); copyto!(C, C0)
    C2 = KernelAbstractions.zeros(backend, eltype(C0), n, n)
    step! = kernel(backend, 256, (n, n))
    for it in 1:nt
        step!(C2, C, γ, dtD)
        C, C2 = C2, C                                # swap: the new C is now in C
    end
    return Array(C)
end

n_check  = 256
nt_check = backend isa CPU ? 2_000 : 20_000
C0 = initial_condition(FT, n_check)
C_two = run_two_kernels(backend, C0, γ, dtD, nt_check)
for (name, kernel) in (("fused", step_fused!), ("fused, interior path", step_fused_interior!))
    C_f = run_fused(kernel, backend, C0, γ, dtD, nt_check)
    err = maximum(abs.(C_f .- C_two))
    @printf("%-22s max |C - C_two_kernels| = %.2e   matches: %s\n", name, err, err < sqrt(eps(FT)))
end

# ---- 2. time per step ----

n = backend isa CPU ? 512 : 8192
C  = KernelAbstractions.allocate(backend, FT, n, n); copyto!(C, initial_condition(FT, n))
C2 = KernelAbstractions.zeros(backend, FT, n, n)
μ  = KernelAbstractions.zeros(backend, FT, n, n)
potential!      = potential_ka!(backend, 256, (n, n))
concentration!  = concentration_ka!(backend, 256, (n, n))
fused!          = step_fused!(backend, 256, (n, n))
fused_interior! = step_fused_interior!(backend, 256, (n, n))

t_two = time_it(() -> (potential!(μ, C, γ); concentration!(C, μ, dtD)), backend)
t_f   = time_it(() -> fused!(C2, C, γ, dtD), backend)            # no swap needed for timing
t_fi  = time_it(() -> fused_interior!(C2, C, γ, dtD), backend)

@printf("\n%d², %s, %s\n", n, nameof(typeof(backend)), FT)
@printf("two kernels (5 arrays):          %8.3f ms per step\n", t_two * 1e3)
@printf("fused (2 arrays):                %8.3f ms per step  (%.2f× faster)\n", t_f * 1e3, t_two / t_f)
@printf("fused, interior path (2 arrays): %8.3f ms per step  (%.2f× faster)\n", t_fi * 1e3, t_two / t_fi)
