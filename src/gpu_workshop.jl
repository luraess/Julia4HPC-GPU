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
# workshop environment and loads the packages we need. On a Mac, the first run prints
# a long list of warnings from AMDGPU, such as `HIP library is unavailable`: ignore
# them, AMDGPU is only used on AMD GPUs.

#src TODO: align with the shared depot on Arctic (Manifest.toml, whether to instantiate here).
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()

using KernelAbstractions
using CairoMakie
using Printf, Random, Statistics

# Now choose the device to run on. The CPU is active by default: it is only there
# to check that the code runs, the performance sections are meant for a GPU.
#
# To use a GPU, comment out the two CPU lines and uncomment the lines of your GPU
# (select them and press `Ctrl + /`). On Arctic, that is the NVIDIA block on the
# `ar_mig` and `ar_a100` partitions, and the AMD block on `ar_mi210`.
#
# `FT` is the floating-point type we compute in. Apple GPUs do not support `Float64`.
#
# > **On an Apple GPU?** The Metal block also loads a small, temporary fix,
# > [`extras/metal_index_fix.jl`](../extras/metal_index_fix.jl). Without it, our
# > kernels run at only about 20% of the memory bandwidth: Metal.jl computes the
# > index of each work-item with 64-bit integer divisions, which Apple GPUs emulate
# > in software ([Metal.jl#910](https://github.com/JuliaGPU/Metal.jl/issues/910)).
# > The fix does the same arithmetic in 32 bits. It will no longer be needed once
# > Metal.jl does this itself.

backend = CPU();  FT = Float64                      # CPU: only to check that the code runs
device_name = Sys.cpu_info()[1].model

## using CUDA                                       # NVIDIA
## backend = CUDABackend();  FT = Float64
## device_name = CUDA.name(CUDA.device())           # which GPU (or MIG slice) you got

## using AMDGPU                                     # AMD
## backend = ROCBackend();  FT = Float64
## device_name = AMDGPU.HIP.name(AMDGPU.device())   # which GPU you got

## using Metal                                      # Apple laptop
## include(joinpath(@__DIR__, "..", "extras", "metal_index_fix.jl"))   # temporary fix, see above
## backend = MetalBackend();  FT = Float32
## device_name = string(Metal.device().name)

println("backend = ", nameof(typeof(backend)), ",  FT = ", FT, ",  device = ", device_name)

# ## 1. Why GPUs
#
# Solving PDEs gets expensive fast. The equation we will solve, Cahn-Hilliard, is
# fourth order in space: with an explicit scheme the time step shrinks as `dt ∝ dx⁴`.
# Halving the grid spacing in 2D means 4× more cells *and* 16× more time steps, so
# **64× more work**.
#
# Since the mid-2000s, processors have stopped getting faster per core. Performance
# now comes from parallelism: more cores, wider vector units. GPUs take this
# furthest, with thousands of lightweight threads. More important for us, they also
# have **much higher memory bandwidth** than CPUs:
#
# Here is the hardware of the Arctic nodes we run on:
#
# | device | FP64 peak [TFLOP/s] | memory bandwidth [GB/s] | balance [flop per number] |
# |:-------|-------:|-------:|-----:|
# | AMD EPYC 7543 (CPU, one socket, 32 cores) | 1.4 | 205 | ~56 |
# | NVIDIA A100 SXM 80 GB | 9.7 | 2039 | ~38 |
# | AMD Instinct MI210 | 22.6 | 1638 | ~110 |
#
# *Vendor peak values. An A100 node has two EPYC 7543 sockets and eight A100s. The
# balance is FLOP/s ÷ (bytes/s) × 8 bytes: how many floating-point operations a
# device can do in the time it takes to load one `Float64` from memory.*
#
# Two things to take from this table:
# - A GPU has about **10× the memory bandwidth** of a CPU socket.
# - Every device can do **40–110 flops per number it loads**. Unless a code does
#   that much arithmetic per number, it waits for memory, not for compute
#   (section 3).
#
# For such memory-bound codes, the speedup of a GPU over a CPU is about the ratio
# of their memory bandwidths.
#
# On Arctic, the `ar_mig` partition gives you a *slice* of an A100 (MIG). A slice
# gets a share of the memory bandwidth: roughly 1/8 for `1g.10gb`, 1/4 for `2g.20gb`
# and 1/2 for `3g.40gb`. We will measure what you actually get in section 5.

# ## 2. The problem: Cahn-Hilliard in 2D
#
# The [Cahn-Hilliard equation](https://en.wikipedia.org/wiki/Cahn%E2%80%93Hilliard_equation)
# describes how a binary mixture, think oil and water, separates into two phases:
#
# ```math
# \frac{\partial C}{\partial t} = D \nabla^2 \mu , \qquad \mu = C^3 - C - \gamma \nabla^2 C
# ```
#
# - `C` is the concentration, between -1 and 1 (the two pure phases).
# - `μ` is the chemical potential, `D` the mobility, and `γ` sets the width of the
#   interface between the phases.
# - The boundaries are closed (no flux), so the amount of each phase is conserved:
#   **`mean(C)` stays constant**.
# - The free energy **`F` decreases** over time.
#
# These last two properties will be our correctness checks.
#
# Starting from small random noise, the phases separate and then coarsen. The
# conserved mean `C̄ = mean(C)` selects the pattern: interwoven bands for `C̄ = 0`
# (first), droplets for `C̄ = 0.4` (second):
#
#nb # <video src="../assets/CahnHilliard2D_C0.mp4" width="400" controls autoplay loop muted></video>
#nb # <video src="../assets/CahnHilliard2D_C04.mp4" width="400" controls autoplay loop muted></video>
#md # **`C̄ = 0`**
#md #
#md # https://github.com/user-attachments/assets/03f8ac66-e961-4302-a263-9443137ceee9
#md #
#md # **`C̄ = 0.4`**
#md #
#md # https://github.com/user-attachments/assets/86cb2deb-0275-4ed3-896e-b6de0f58d819
#
# What matters for performance is that each time step makes **two passes** over the
# grid:
# 1. compute `μ` from `C`, which needs the neighbours of `C` for `∇²C`;
# 2. update `C` from `μ`, which needs the neighbours of `μ` for `∇²μ`.
#
# Since pass 2 needs `μ` at the neighbouring cells, we store `μ` in an array between
# the two passes. Each time step therefore reads `C` and writes `μ`, then reads `μ`
# and `C` and writes `C`: **5 array accesses** per time step.

# ## 3. What limits performance
#
# A computation is limited either by how fast the device does arithmetic
# (**compute bound**), or by how fast it moves data to and from memory
# (**memory bound**).
#
# Let's count for pass 1, `μ = C³ - C - γ∇²C`, at one grid cell, with `c = C[ix, iy]`:
#
# ```julia
# μ[ix, iy] = c*c*c - c - γ * (C[ix-1, iy] + C[ix+1, iy] + C[ix, iy-1] + C[ix, iy+1] - 4c)
# ```
#
# - **Memory:** read `C`, write `μ`: 2 numbers, 16 bytes. The neighbours of `C`
#   were already loaded for the neighbouring cells, so they come from cache.
# - **Arithmetic:** about 10 floating-point operations.
#
# That is ~5 flops per number moved, where the devices above could do 40–110. The
# GPU spends most of its time waiting for memory: **flops are (almost) free, bytes
# are expensive**.
#
# So FLOP/s is the wrong metric for this kind of code. What we need to measure is
# how fast we move memory.

# ## 4. Memory bound: memcopy is the ceiling
#
# For memory-bound codes we measure the **effective memory throughput**
#
# ```math
# T_\mathrm{eff} = \frac{n_\mathrm{arrays} \cdot n_x \cdot n_y \cdot \mathrm{sizeof(FT)}}{10^9 \cdot t} \quad [\mathrm{GB/s}]
# ```
#
# where `sizeof(FT)` is the size of one number in bytes (8 for `Float64`), `t` is
# the time of one call (or one time step) in seconds, and `n_arrays` counts the
# arrays that **must** be read or written, assuming neighbours come from cache:
#
# | operation | arrays moved |
# |:----------|:-------------|
# | memcopy `A = B` | 2: read `B`, write `A` |
# | Cahn-Hilliard, one time step | 5: read `C`, write `μ`, read `μ`, read `C`, write `C` |
#
# For Cahn-Hilliard, 5 is the minimum for *our two-pass algorithm*, which stores `μ`
# between the passes (section 2).
#
# A memcopy does nothing but move data, so no code moving the same amount of data
# can be faster. **The memcopy throughput measured on your GPU, `T_peak`, is our
# ceiling**, and we will report every implementation as a fraction of it.
#
# Why measure it rather than take the vendor's number? Vendor peaks are not reached
# in practice (80–90% of them is typical), and on a MIG slice you only get part of
# the GPU. Measuring tells us what *your* device can actually do.

# ## 5. Measuring memcopy without KernelAbstractions
#
# KernelAbstractions gives us a portable way to allocate arrays on the device:
# `KernelAbstractions.allocate(backend, FT, nx, ny)` (uninitialised) and
# `KernelAbstractions.zeros(backend, FT, nx, ny)`. On NVIDIA they return a
# `CuArray`, on AMD a `ROCArray`, on the CPU a plain `Array`:

A_small = KernelAbstractions.zeros(backend, FT, 4, 4)
typeof(A_small)

# In this section we only use KernelAbstractions to get the arrays. The copies are
# done without writing any kernel ourselves: with `copyto!(A, B)`, the vendor's
# tuned copy, and with broadcasting, `A .= B`, for which Julia generates a GPU
# kernel for us.
#
# ### GPU operations are asynchronous
#
# Launching work on a GPU returns immediately, before the work is done. To time it,
# we must wait for the GPU to finish with `KernelAbstractions.synchronize(backend)`:

n = 8192
haskey(ENV, "CI") && (n = 512)                 # CI only: small size, fast run  #src
A = KernelAbstractions.zeros(backend, FT, n, n)
B = KernelAbstractions.allocate(backend, FT, n, n)
copyto!(B, rand(FT, n, n))                 # random values, copied from the CPU

A .= B                                     # the first call compiles the broadcast kernel
KernelAbstractions.synchronize(backend)

t0 = time()
A .= B
t_launch = time() - t0                     # time to *launch* the copy
KernelAbstractions.synchronize(backend)

t0 = time()
A .= B
KernelAbstractions.synchronize(backend)    # wait until the GPU has finished
t_done = time() - t0                       # time to *do* the copy

@printf("without synchronize: %8.3f ms\n", t_launch * 1e3)
@printf("with    synchronize: %8.3f ms\n", t_done * 1e3)

# On the CPU both numbers are about the same: there, everything is synchronous.
#
# ### Timing and T_eff
#
# Two small helpers. `time_it` runs `f` once to compile and warm up, then times
# `nrep` calls and keeps the best of `ntrial` trials, to filter out noise from
# other users on the machine:

function time_it(f, backend; nrep=20, ntrial=3)
    f()                                          # warm-up: the first call also compiles
    KernelAbstractions.synchronize(backend)
    t_best = Inf
    for trial in 1:ntrial
        t0 = time()
        for rep in 1:nrep
            f()
        end
        KernelAbstractions.synchronize(backend)  # wait for the GPU before stopping the clock
        t_best = min(t_best, (time() - t0) / nrep)
    end
    return t_best                                # seconds per call
end

# `T_eff` follows the formula of section 4, in GB/s, for arrays the size of `A`:

T_eff(narrays, A, t) = narrays * length(A) * sizeof(eltype(A)) / t / 1e9

# Back to our 8192 × 8192 copy, which moves 2 arrays:

t = time_it(() -> copyto!(A, B), backend)
@printf("copyto!: t = %.3f ms,  T_eff = %.1f GB/s\n", t * 1e3, T_eff(2, A, t))

# ### Size matters
#
# Let's repeat this for several array sizes. Note that the loop lives inside a
# function: in Julia, code in functions is compiled and fast, and the variables
# inside do not clash with the ones of the notebook.

function memcopy_sweep(backend, FT, ns)
    T_copyto    = Float64[]
    T_broadcast = Float64[]
    for n in ns
        A = KernelAbstractions.zeros(backend, FT, n, n)
        B = KernelAbstractions.allocate(backend, FT, n, n)
        copyto!(B, rand(FT, n, n))
        t_copyto    = time_it(() -> copyto!(A, B), backend)
        t_broadcast = time_it(() -> (A .= B), backend)
        push!(T_copyto,    T_eff(2, A, t_copyto))
        push!(T_broadcast, T_eff(2, A, t_broadcast))
        @printf("n = %5d   copyto!: %7.1f GB/s   A .= B: %7.1f GB/s\n",
                n, T_copyto[end], T_broadcast[end])
    end
    return T_copyto, T_broadcast
end

ns = [512, 1024, 2048, 4096, 8192]
haskey(ENV, "CI") && (ns = [256, 512])         # CI only: small sizes, fast run  #src
T_copyto, T_broadcast = memcopy_sweep(backend, FT, ns)

#-

fig = Figure(size=(600, 400))
ax  = Axis(fig[1, 1]; xscale=log2, xticks=ns, xlabel="n  (arrays of n × n)",
           ylabel="T_eff [GB/s]", title="memcopy, $(nameof(typeof(backend))), $FT")
scatterlines!(ax, ns, T_copyto;    label="copyto!")
scatterlines!(ax, ns, T_broadcast; label="A .= B")
axislegend(ax; position=:lt)
fig

# Small arrays do not measure the memory bandwidth: the launch overhead dominates,
# or the arrays fit in the GPU's cache. Only the largest sizes tell us what the
# memory can do. As a first estimate of our ceiling, we take the vendor copy at the
# largest size:

T_peak = T_copyto[end]
@printf("T_peak = %.1f GB/s\n", T_peak)

# How does your `T_peak` compare to the vendor number in section 1, or to your
# share of it on a MIG slice?

# ## 6. A first kernel: memcopy with KernelAbstractions
#
# Now we write the copy ourselves. On the CPU, a 2D copy is a double loop:
#
# ```julia
# for iy in 1:ny, ix in 1:nx
#     A[ix, iy] = B[ix, iy]
# end
# ```
#
# A GPU kernel keeps only the **body of the loop**: it describes what happens at one
# `(ix, iy)`. The GPU then runs it for all `(ix, iy)` at once, one *work-item* (a
# *thread* in CUDA terms) each. With KernelAbstractions:
#
# - `@kernel` turns a function into a kernel. A kernel returns nothing, it writes
#   into its arguments.
# - `@index(Global, NTuple)` gives the work-item its `(ix, iy)`.
# - `inbounds = true` switches off bounds checking inside the kernel. We will make
#   sure that every `(ix, iy)` is inside the arrays.

@kernel inbounds = true function memcopy_ka!(A, B)
    ix, iy = @index(Global, NTuple)  #sol
    #hint ix, iy = ???               # ask KernelAbstractions for this work-item's (ix, iy)
    A[ix, iy] = B[ix, iy]
end

# Running a kernel takes two steps. First we *instantiate* it for our backend, then
# we *launch* it, giving the `ndrange`: the range of `(ix, iy)` to run over. Here
# we want one work-item per array element:

memcopy_dyn! = memcopy_ka!(backend)          # 1. instantiate for our backend
memcopy_dyn!(A, B; ndrange = size(A))        # 2. launch over all (ix, iy)  #sol
#hint memcopy_dyn!(A, B; ndrange = ???)            # 2. launch over all (ix, iy)
KernelAbstractions.synchronize(backend)      # 3. wait for the GPU to finish

@assert Array(A) == Array(B)                 # copy back to the CPU and compare
println("memcopy_ka! works")

# ### Static or dynamic launch
#
# Above, the sizes were only given at launch: the kernel is *dynamic*. We can also
# fix them when instantiating: the *workgroup size*, i.e. how many work-items are
# grouped together on the GPU (a *thread block* in CUDA; 256 is a good default on
# all GPUs), and the `ndrange`. The compiler then knows them in advance:

memcopy_static! = memcopy_ka!(backend, 256, size(A))   # workgroup size and ndrange fixed
memcopy_static!(A, B)                                  # no ndrange needed at launch

t_dyn    = time_it(() -> memcopy_dyn!(A, B; ndrange = size(A)), backend)
t_static = time_it(() -> memcopy_static!(A, B), backend)

@printf("dynamic: %7.1f GB/s\n", T_eff(2, A, t_dyn))
@printf("static:  %7.1f GB/s\n", T_eff(2, A, t_static))
@printf("copyto!: %7.1f GB/s\n", T_copyto[end])

# The static kernel is usually faster: knowing the sizes lets the compiler simplify
# how each work-item computes its `(ix, iy)`. Section 10 looks at what `@index`
# actually does. From now on, we always use static kernels.
#
# ### How close to the ceiling?
#
# Let's sweep the sizes again with the static kernel. **Exercise:** how many arrays
# does a memcopy move?

function memcopy_ka_sweep(backend, FT, ns)
    T_ka = Float64[]
    for n in ns
        A = KernelAbstractions.zeros(backend, FT, n, n)
        B = KernelAbstractions.allocate(backend, FT, n, n)
        copyto!(B, rand(FT, n, n))
        memcopy! = memcopy_ka!(backend, 256, (n, n))
        t = time_it(() -> memcopy!(A, B), backend)
        push!(T_ka, T_eff(2, A, t))  #sol
        #hint push!(T_ka, T_eff(???, A, t))
        @printf("n = %5d   KA memcopy: %7.1f GB/s\n", n, T_ka[end])
    end
    return T_ka
end

T_ka = memcopy_ka_sweep(backend, FT, ns)

#-

T_memcopy_ka = T_ka[end]
@printf("KA memcopy: %.1f GB/s = %.0f%% of copyto!\n", T_memcopy_ka, 100 * T_memcopy_ka / T_copyto[end])

# A few lines of portable Julia get close to the vendor's tuned copy, and on some
# GPUs (AMD MI200 series, for example) they even beat it. The vendor copy is not
# always the fastest way to copy, so **our ceiling is the fastest memcopy we
# measured**:

T_peak = max(T_copyto[end], T_broadcast[end], T_memcopy_ka)
@printf("T_peak = %.1f GB/s\n", T_peak)

#-

fig = Figure(size=(600, 400))
ax  = Axis(fig[1, 1]; xscale=log2, xticks=ns, xlabel="n  (arrays of n × n)",
           ylabel="T_eff [GB/s]", title="memcopy, $(nameof(typeof(backend))), $FT")
scatterlines!(ax, ns, T_copyto;    label="copyto!")
scatterlines!(ax, ns, T_broadcast; label="A .= B")
scatterlines!(ax, ns, T_ka;        label="KA kernel")
hlines!(ax, T_peak; color=:gray, linestyle=:dash, label="T_peak")
axislegend(ax; position=:lt)
fig

# `T_peak` is the baseline for the Cahn-Hilliard kernels: they cannot beat it, and
# we will see how close they get.
#
# ### Share your results
#
# Copy the line printed below into the results form (link given during the
# workshop). It holds your device and the three memcopy throughputs, in GB/s:
# `copyto!`, `A .= B` and the KA kernel. We will compare the GPUs of the whole room.

println(join((device_name, nameof(typeof(backend)), FT, round(T_copyto[end]; digits=1),
              round(T_broadcast[end]; digits=1), round(T_memcopy_ka; digits=1)), "; "))

# ## 7. Cahn-Hilliard: discretisation and CPU reference
#
# Before going to the GPU, we write the solver in plain Julia, with loops, on the
# CPU. This shows the physics, and gives us a **reference solution** to check the
# GPU versions against.
#
# ### Discretisation
#
# **Grid units.** We measure lengths in grid cells: `dx = dy = 1`. All constants are
# then of order one and do not depend on the resolution. This also keeps `Float32`
# accurate enough on Apple GPUs.
#
# **Laplacian.** With `dx = dy = 1`, the 5-point stencil at `(ix, iy)` is
#
# ```
# ∇²A ≈ (A[ix-1, iy] - 2A[ix, iy] + A[ix+1, iy]) + (A[ix, iy-1] - 2A[ix, iy] + A[ix, iy+1])
# ```
#
# **No-flux boundaries.** At the boundary, one neighbour lies outside the array. We
# replace it by the cell itself, as in a mirror: no gradient across the boundary,
# hence no flux. With `min` and `max` the index never leaves the array: at the right
# boundary, `A[min(ix+1, nx), iy]` is `A[nx, iy]`. No `if` is needed, which GPUs like.
#
# **Time step.** We step explicitly in time: compute `μ` everywhere, then update `C`
# everywhere. This is stable for `dt ≤ 2 / (D κ (γκ + 2))`, where `κ = 8` is the
# largest eigenvalue of `-∇²` on the grid (`4/dx² + 4/dy²`). We use half of that.

D     = FT(1)                                  # mobility
wcell = FT(4)                                  # interface width, in cells
γ     = wcell^2 / 8                            # gradient energy coefficient
κmax  = FT(8)                                  # largest eigenvalue of -∇²: 4/dx² + 4/dy²
dt    = 2 / (D * κmax * (γ * κmax + 2)) / 2    # half the explicit stability limit
@printf("γ = %.3g,  dt = %.3g\n", γ, dt)

# The initial condition is small random noise around a mean `C̄`. A fixed seed gives
# the same field every time, so the GPU versions can start from exactly the same
# state:

function initial_condition(FT, n; C̄=0, ampl=0.02, seed=1234)
    Random.seed!(seed)
    C = C̄ .+ FT(ampl) .* randn(FT, n, n)
    C .+= C̄ - mean(C)                          # pin the mean to exactly C̄
    return C
end

# ### The solver
#
# A word on sizes: all our grids are square, `n × n` cells, and `n` is the size we
# choose for a run or a benchmark. The functions that compute on arrays read
# `nx, ny = size(A)` instead, so they work for rectangular grids too.
#
# The Laplacian at `(ix, iy)`, with the mirror boundaries. `Base.@propagate_inbounds`
# lets the `@inbounds` of the caller apply inside `lap` as well (more in section 10).
#
# **Exercise:** complete the y-direction, following the x-direction.

Base.@propagate_inbounds function lap(A, ix, iy, nx, ny)
    a = A[ix, iy]
    return (A[max(ix-1, 1), iy] - 2a + A[min(ix+1, nx), iy]) +   # x-direction
           (A[ix, max(iy-1, 1)] - 2a + A[ix, min(iy+1, ny)])     # y-direction  #sol
    #hint        ???                                                   # y-direction
end

# Pass 1 computes the chemical potential `μ = C³ - C - γ∇²C` at every cell. Julia
# arrays are stored column by column, so the first index, `ix`, is the inner loop.
#
# **Exercise:** write the update of `μ[ix, iy]`.

function chemical_potential!(μ, C, γ)
    nx, ny = size(C)
    @inbounds for iy in 1:ny, ix in 1:nx
        c = C[ix, iy]
        μ[ix, iy] = c^3 - c - γ * lap(C, ix, iy, nx, ny)  #sol
        #hint μ[ix, iy] = ???
    end
    return
end

# Pass 2 updates the concentration, `C = C + dt·D·∇²μ`. We pass `dtD = dt * D` as
# a single number.
#
# **Exercise:** write the update of `C[ix, iy]`.

function update_concentration!(C, μ, dtD)
    nx, ny = size(C)
    @inbounds for iy in 1:ny, ix in 1:nx
        C[ix, iy] += dtD * lap(μ, ix, iy, nx, ny)  #sol
        #hint C[ix, iy] += ???
    end
    return
end

# The free energy, `F = Σ (C² - 1)²/4 + γ/2 |∇C|²`, which must decrease over time.
# The gradient terms are differences between neighbouring cells:

function free_energy(C, γ)
    nx, ny = size(C)
    F = zero(eltype(C))
    for iy in 1:ny, ix in 1:nx
        c = C[ix, iy]
        F += (c^2 - 1)^2 / 4
        if ix < nx
            F += γ / 2 * (C[ix+1, iy] - c)^2
        end
        if iy < ny
            F += γ / 2 * (C[ix, iy+1] - c)^2
        end
    end
    return F
end

# The time loop. Every `nout` steps it keeps a snapshot of `C` and the free energy,
# for plotting:

function cahn_hilliard_cpu(C0, γ, dtD, nt; nout=nt)
    C  = copy(C0)
    μ  = zeros(eltype(C), size(C))
    Cs = [copy(C)]                             # snapshots of C, every nout steps
    Fs = [free_energy(C, γ)]                   # free energy, every nout steps
    for it in 1:nt
        chemical_potential!(μ, C, γ)           # pass 1
        update_concentration!(C, μ, dtD)       # pass 2
        if it % nout == 0
            push!(Cs, copy(C))
            push!(Fs, free_energy(C, γ))
        end
    end
    return C, Cs, Fs
end

# ### Run it
#
# A small grid, since plain loops on one CPU core are slow:

n    = 256
nt   = 20_000
haskey(ENV, "CI") && (nt = 2_000)              # CI only: fewer steps, fast run  #src
nout = 500
C0   = initial_condition(FT, n)

cahn_hilliard_cpu(C0, γ, dt * D, 10)           # warm-up: compiles the functions
t0 = time()
C_ref, Cs, Fs = cahn_hilliard_cpu(C0, γ, dt * D, nt; nout)
t_cpu = time() - t0
@printf("CPU reference: %d steps in %.1f s,  T_eff = %.1f GB/s\n",
        nt, t_cpu, T_eff(5, C0, t_cpu / nt))

# Did it work? The mean must stay constant, and the free energy must decrease:

@printf("mean(C): %+.2e -> %+.2e   (must stay constant)\n", mean(C0), mean(C_ref))
@printf("F:       %.6g -> %.6g   (must decrease)\n", Fs[1], Fs[end])
println("F decreases at every output: ", all(diff(Fs) .< 0))

# The movie of the run:

ts  = (0:length(Cs)-1) .* (nout * dt)          # time of each snapshot
fig = Figure(size=(500, 450))
ax  = Axis(fig[1, 1]; aspect=DataAspect(), xlabel="x", ylabel="y")
hm  = heatmap!(ax, Cs[1]; colormap=:balance, colorrange=(-1, 1))
Colorbar(fig[1, 2], hm; label="C")
Record(fig, eachindex(Cs); framerate=8) do i
    hm[1] = Cs[i]
    ax.title = @sprintf("t = %.1f", ts[i])
end

#-

lines(ts, Fs; axis=(xlabel="t", ylabel="free energy F"))

# `C_ref`, the field after `nt` steps from `C0`, is our reference. In sections 8
# and 10 we run the same steps on the GPU and compare with it.

# ## 8. Cahn-Hilliard with array programming
#
# > **New session, or restarted the kernel?** The sections below use results from
# > above, such as `T_peak` and the reference solution `C_ref`. Select this cell and
# > run *Run → Run All Above Selected Cell*: it takes a minute or two on a GPU.
#
# Our first GPU version uses **array programming**: we write operations on whole
# arrays with broadcasting (the `.`), and no loops, as in NumPy or Matlab. Each
# broadcast line, such as `A .= B .+ C`, becomes one GPU kernel that Julia
# generates for us. The arrays are allocated on the device with KernelAbstractions,
# so the same code runs on any backend.
#
# ### The Laplacian with whole arrays
#
# To write `∇²A` with whole arrays, we split it in two steps, `∇²A = ∂/∂x(∂A/∂x) + ∂/∂y(∂A/∂y)`:
#
# 1. The differences between neighbouring cells live on the **faces** between the
#    cells: `qx[ix, iy] = A[ix, iy] - A[ix-1, iy]` on the face left of cell `ix`.
#    With the two boundary faces, there are `nx + 1` faces along x.
# 2. The Laplacian in a cell is the difference of its two face values:
#    `qx[ix+1, iy] - qx[ix, iy]`, plus the same along y.
#
# ```
#  faces:   qx[1]    qx[2]    qx[3]   ...   qx[nx]   qx[nx+1]
#             |  A[1]  |  A[2]  |      ...     |  A[nx]  |
#             0                                          0     <- boundary faces
# ```
#
# **No-flux boundaries** come for free: the two boundary faces are allocated with
# zeros and never written, so nothing flows through them. This is exactly the mirror
# of section 7: at `ix = 1`, `qx[2] - qx[1] = A[2] - A[1] = A[2] - 2A[1] + A[1]`.
#
# `@views` in front of a function makes every `A[2:nx, :]` inside it a *view* into
# `A` rather than a copy: no extra memory, and no extra memory traffic.
#
# **Exercise:** complete the y-direction, following the x-direction.

@views function gradient!(qx, qy, A)
    nx, ny = size(A)
    qx[2:nx, :] .= A[2:nx, :] .- A[1:nx-1, :]      # inner x-faces
    qy[:, 2:ny] .= A[:, 2:ny] .- A[:, 1:ny-1]      # inner y-faces  #sol
    #hint qy[:, 2:ny] .= ???                             # inner y-faces
    return
end

# The two updates, one broadcast each. Once `gradient!(qx, qy, A)` has run, the
# Laplacian of `A` is `qx[2:nx+1, :] .- qx[1:nx, :] .+ qy[:, 2:ny+1] .- qy[:, 1:ny]`.
#
# **Exercise:** write `μ = C³ - C - γ∇²C` and `C = C + dt·D·∇²μ` as broadcasts.

@views function potential_ap!(μ, C, qx, qy, γ)
    nx, ny = size(C)
    μ .= C.^3 .- C .- γ .* (qx[2:nx+1, :] .- qx[1:nx, :] .+ qy[:, 2:ny+1] .- qy[:, 1:ny])  #sol
    #hint μ .= ???
    return
end

@views function concentration_ap!(C, qx, qy, dtD)
    nx, ny = size(C)
    C .+= dtD .* (qx[2:nx+1, :] .- qx[1:nx, :] .+ qy[:, 2:ny+1] .- qy[:, 1:ny])  #sol
    #hint C .+= ???
    return
end

# One time step makes the same two passes as in section 7:

function step_ap!(C, μ, qx, qy, γ, dtD)
    gradient!(qx, qy, C)                       # pass 1: ∇C on the faces,
    potential_ap!(μ, C, qx, qy, γ)             #         then μ
    gradient!(qx, qy, μ)                       # pass 2: ∇μ on the faces,
    concentration_ap!(C, qx, qy, dtD)          #         then C
    return
end

# The solver allocates the arrays on the device, uploads the initial condition, runs
# the time loop, and returns the result as a CPU `Array`:

function cahn_hilliard_ap(backend, C0, γ, dtD, nt)
    nx, ny = size(C0)
    FT = eltype(C0)
    C  = KernelAbstractions.allocate(backend, FT, nx, ny)
    copyto!(C, C0)                                     # upload the initial condition
    μ  = KernelAbstractions.zeros(backend, FT, nx, ny)
    qx = KernelAbstractions.zeros(backend, FT, nx + 1, ny)   # x-faces, boundary faces stay 0
    qy = KernelAbstractions.zeros(backend, FT, nx, ny + 1)   # y-faces, boundary faces stay 0
    for it in 1:nt
        step_ap!(C, μ, qx, qy, γ, dtD)
    end
    return Array(C)                                    # download the result
end

# ### Run it and compare with the reference
#
# The same initial condition and number of steps as the CPU reference:

C_ap = cahn_hilliard_ap(backend, C0, γ, dt * D, nt)

err_ap = maximum(abs.(C_ap .- C_ref))
@printf("max |C_ap - C_ref| = %.2e\n", err_ap)
@printf("mean(C) = %+.2e,  F = %.6g  (reference: %.6g)\n",
        mean(C_ap), free_energy(C_ap, γ), Fs[end])
println("matches the reference: ", err_ap < sqrt(eps(FT)))

# The two versions do the same arithmetic in a different order, so they agree to
# round-off rather than bit for bit: about `1e-14` in `Float64` (`1e-5` in
# `Float32`) after 20 000 steps. We accept differences below `sqrt(eps(FT))`.

# ## 9. Performance of array programming
#
# How fast is the array version? We time each piece of `step_ap!`, and the whole
# step, on a large grid: the same size as for the memcopy, so we can compare with
# `T_peak`.
#
# For `T_eff`, we need the number of arrays each piece moves. As before, neighbours
# come from cache, and an array that is read and written counts twice. A face array
# counts as one array: it is only one row or column longer than the grid.
#
# **Exercise:** fill in how many arrays each piece moves. Careful: `gradient!` runs
# two kernels, one per direction.

function bench_ap(backend, FT, n, γ, dtD)
    C  = KernelAbstractions.allocate(backend, FT, n, n)
    copyto!(C, initial_condition(FT, n))       # realistic data: small noise
    μ  = KernelAbstractions.zeros(backend, FT, n, n)
    qx = KernelAbstractions.zeros(backend, FT, n + 1, n)
    qy = KernelAbstractions.zeros(backend, FT, n, n + 1)
    ## time each piece, then the whole step
    t_grad = time_it(() -> gradient!(qx, qy, C), backend)
    t_pot  = time_it(() -> potential_ap!(μ, C, qx, qy, γ), backend)
    t_con  = time_it(() -> concentration_ap!(C, qx, qy, dtD), backend)
    t_step = time_it(() -> step_ap!(C, μ, qx, qy, γ, dtD), backend)
    ## arrays each piece moves
    narr_grad = 4  #sol
    #hint narr_grad = ???
    narr_pot  = 4  #sol
    #hint narr_pot  = ???
    narr_con  = 4  #sol
    #hint narr_con  = ???
    narr_step = 5                              # the minimum for one step (section 4)
    return [T_eff(narr_grad, C, t_grad), T_eff(narr_pot, C, t_pot),
            T_eff(narr_con, C, t_con),   T_eff(narr_step, C, t_step)]
end

T_ap     = bench_ap(backend, FT, ns[end], γ, dt * D)
T_eff_ap = T_ap[end]                           # the whole step, kept for the final comparison
labels   = ["gradient!", "potential_ap!", "concentration_ap!", "whole step"]
for (label, T) in zip(labels, T_ap)
    @printf("%-18s %7.1f GB/s   %3.0f%% of T_peak\n", label, T, 100 * T / T_peak)
end

#-

fig = Figure(size=(600, 400))
ax  = Axis(fig[1, 1]; xticks=(1:4, labels), ylabel="T_eff [GB/s]",
           title="array programming, $(nameof(typeof(backend))), $FT")
barplot!(ax, 1:4, T_ap)
hlines!(ax, T_peak; color=:gray, linestyle=:dash, label="T_peak")
axislegend(ax; position=:rt)
fig

# Each piece runs at a good fraction of `T_peak`: broadcasting generates efficient
# kernels. The whole step does not. Its `T_eff` counts the 5 arrays a time step of
# our two-pass algorithm *must* move, but array programming moves many more:
#
# | piece | arrays moved |
# |:------|-------------:|
# | `gradient!(qx, qy, C)` | 4 |
# | `potential_ap!` | 4 |
# | `gradient!(qx, qy, μ)` | 4 |
# | `concentration_ap!` | 4 |
# | **one time step** | **16**, where 5 would do |
#
# So even with every piece at memcopy speed, the step reaches at most 5/16, about
# 30% of `T_peak`. The face arrays `qx` and `qy` are intermediate results that make
# a round trip through memory, and `C` and `μ` are read several times.
#
# To do better, we must fuse the work into fewer kernels that keep intermediate
# values in registers, close to the compute units. This is what kernel programming
# gives us.
#
# > **Take-away: array programming on the GPU runs one kernel per statement.**
# > Julia fuses all the dots of one statement, such as
# > `μ .= C.^3 .- C .- γ .* (...)`, into a single kernel. Separate statements are
# > separate kernels, and each one reads its inputs from memory and writes its
# > result back to memory. Easy to write and correct, but every intermediate array
# > costs memory traffic.

# ## 10. Kernel programming with neighbours
#
# With kernels we write ourselves, each pass of the time step becomes **one**
# kernel. Everything in between (`∇²C`, `∇²μ`, the differences between neighbours)
# stays in registers. A time step then moves only the 5 arrays our algorithm needs,
# instead of 16.
#
# The recipe is the one of section 6: **the kernel body is the body of the loop**.
# The neighbours are read with the very same `lap` function as in section 7, mirror
# boundaries included: one function serves the CPU loops and the GPU kernels.
#
# **Exercise:** complete the two kernels, using the bodies of your loops in
# `chemical_potential!` and `update_concentration!` (section 7).

@kernel inbounds = true function potential_ka!(μ, C, γ)
    ix, iy = @index(Global, NTuple)
    nx, ny = size(C)
    c = C[ix, iy]
    μ[ix, iy] = c^3 - c - γ * lap(C, ix, iy, nx, ny)  #sol
    #hint μ[ix, iy] = ???
end

@kernel inbounds = true function concentration_ka!(C, μ, dtD)
    ix, iy = @index(Global, NTuple)
    nx, ny = size(C)
    C[ix, iy] += dtD * lap(μ, ix, iy, nx, ny)  #sol
    #hint C[ix, iy] += ???
end

# The solver has the same structure as the array version. The two kernels are
# instantiated once, as static kernels, before the time loop:

function cahn_hilliard_ka(backend, C0, γ, dtD, nt)
    nx, ny = size(C0)
    FT = eltype(C0)
    C  = KernelAbstractions.allocate(backend, FT, nx, ny)
    copyto!(C, C0)                                     # upload the initial condition
    μ  = KernelAbstractions.zeros(backend, FT, nx, ny)
    potential!     = potential_ka!(backend, 256, (nx, ny))
    concentration! = concentration_ka!(backend, 256, (nx, ny))
    for it in 1:nt
        potential!(μ, C, γ)                            # pass 1
        concentration!(C, μ, dtD)                      # pass 2
    end
    KernelAbstractions.synchronize(backend)
    return Array(C)                                    # download the result
end

# ### Run it and compare with the reference

C_ka = cahn_hilliard_ka(backend, C0, γ, dt * D, nt)

err_ka = maximum(abs.(C_ka .- C_ref))
@printf("max |C_ka - C_ref| = %.2e\n", err_ka)
@printf("mean(C) = %+.2e,  F = %.6g  (reference: %.6g)\n",
        mean(C_ka), free_energy(C_ka, γ), Fs[end])
println("matches the reference: ", err_ka < sqrt(eps(FT)))

# The kernels do the same arithmetic in the same order as the CPU loops, so on many
# GPUs the result is identical to the reference, bit for bit.
#
# ### Going further (read later)
#
# The rest of this section explains what happens under the hood. It is not needed
# for section 11.
#
# #### What `@index` does
#
# KernelAbstractions splits the `ndrange` into **workgroups** of `workgroupsize`
# work-items. The GPU runs each workgroup on one of its compute units (an SM on
# NVIDIA, a CU on AMD), in bunches of 32 (NVIDIA) or 64 (AMD) work-items that execute
# in lockstep.
#
# Under the hood, all backends launch a flat list of workgroups, each a flat list of
# work-items. The hardware gives every work-item two integers: the number of its
# workgroup, and its number inside the workgroup. `@index(Global, NTuple)` turns
# these into `(ix, iy)`. A 2D workgroup size such as `(8, 4)` is only the *shape of
# the tile* of the array that a workgroup covers.
#
# Each vendor has its own names for the same things. CUDA's are the most common in
# GPU programming:
#
# | KernelAbstractions | CUDA | AMD (HIP) | Apple (Metal) |
# |:-------------------|:-----|:----------|:--------------|
# | work-item | thread | thread (work-item) | thread |
# | workgroup | thread block | block (workgroup) | threadgroup |
# | all work-items of a launch | grid | grid | grid |
# | `workgroupsize` | threads per block | block size | threads per threadgroup |
# | `@index(Local, …)` | `threadIdx` | `threadIdx` | `thread_position_in_threadgroup` |
# | `@index(Group, …)` | `blockIdx` | `blockIdx` | `threadgroup_position_in_grid` |
# | `@index(Global, …)` | `(blockIdx-1) * blockDim + threadIdx` | same as CUDA | `thread_position_in_grid` |
# | lockstep bunch | warp (32) | wavefront (64) | SIMD-group (32) |
# | compute unit | SM (streaming multiprocessor) | CU (compute unit) | GPU core |
#
# In CUDA.jl and AMDGPU.jl these indices start at 1, hence the `-1`. One more
# difference to keep in mind: KernelAbstractions' `ndrange` counts work-items, while
# a CUDA launch takes the number of *blocks*, `cld(ndrange, workgroupsize)`.
#
# Let's look at both numbers for every element of a small 32 × 16 array:

@kernel function index_demo!(group, item)
    ix, iy = @index(Global, NTuple)
    g = @index(Group, Linear)                  # the workgroup of this work-item
    l = @index(Local, Linear)                  # its number inside the workgroup
    group[ix, iy] = g
    item[ix, iy]  = l
end

function index_figure(backend, workgroupsize)
    nx, ny = 32, 16
    group = KernelAbstractions.zeros(backend, Float32, nx, ny)
    item  = KernelAbstractions.zeros(backend, Float32, nx, ny)
    index_demo!(backend, workgroupsize, (nx, ny))(group, item)
    KernelAbstractions.synchronize(backend)
    fig = Figure(size=(700, 260))
    ax1 = Axis(fig[1, 1]; title="@index(Group, Linear)", aspect=DataAspect(), xlabel="ix", ylabel="iy")
    ax2 = Axis(fig[1, 2]; title="@index(Local, Linear)", aspect=DataAspect(), xlabel="ix", ylabel="iy")
    heatmap!(ax1, Array(group); colormap=:tab20)
    heatmap!(ax2, Array(item); colormap=:viridis)
    Label(fig[0, :], "workgroup size $workgroupsize"; font=:bold)
    return fig
end

index_figure(backend, (32, 1))

#-

index_figure(backend, (8, 4))

# With `(32, 1)` each workgroup covers one row segment; with `(8, 4)`, a tile of
# 8 × 4 cells. In both cases, consecutive work-items run along `ix`, the first
# index, along which the array is contiguous in memory. Neighbouring work-items then
# read neighbouring memory addresses, which the GPU combines into few, wide memory
# transactions. For our stencils the tile shape changes little: 256 work-items
# along `ix` is a good default.
#
# #### Why static sizes help
#
# Turning the two integers into `(ix, iy)` takes an integer division and a remainder,
# and GPUs have no fast integer division. With a static kernel (workgroup size and
# `ndrange` fixed when it is instantiated), the divisors are known to the compiler,
# which replaces the divisions by cheap multiplications or bit shifts. That is the
# gain we saw in section 6. Note that `@index(Global, Linear)` is not cheaper:
# KernelAbstractions computes the 2D index first, then converts it back.
#
# On Apple GPUs, Metal.jl does these divisions in 64-bit integers, which the GPU
# emulates in software: that is what the fix loaded in the setup,
# [`extras/metal_index_fix.jl`](../extras/metal_index_fix.jl), changes to 32 bits.
#
# #### Boundaries without `if`
#
# `lap` handles the boundaries with `min` and `max` instead of `if`. The work-items
# of a bunch execute the same instruction at the same time: if only some of them
# take an `if` branch, the bunch runs both branches, one after the other. With `min`
# and `max`, all work-items run the same instructions.
#
# #### `inbounds = true` and `Base.@propagate_inbounds`
#
# `inbounds = true` puts `@inbounds` on the kernel body: no bounds checks on its
# array accesses. It does not reach into the functions the kernel calls. That is
# why `lap` is marked `Base.@propagate_inbounds`, which lets it inherit the
# `@inbounds` of its caller. Without it, every neighbour access in `lap` is
# bounds-checked, which costs about 10% on these stencils.

# ## 11. Performance of kernel programming
#
# Time for the measurements, with the same recipe as in section 9: each kernel, then
# the whole step, on the large grid.
#
# **Exercise:** fill in how many arrays each kernel moves.

function bench_ka(backend, FT, n, γ, dtD)
    C = KernelAbstractions.allocate(backend, FT, n, n)
    copyto!(C, initial_condition(FT, n))       # realistic data: small noise
    μ = KernelAbstractions.zeros(backend, FT, n, n)
    potential!     = potential_ka!(backend, 256, (n, n))
    concentration! = concentration_ka!(backend, 256, (n, n))
    ## time each kernel, then the whole step
    t_pot  = time_it(() -> potential!(μ, C, γ), backend)
    t_con  = time_it(() -> concentration!(C, μ, dtD), backend)
    t_step = time_it(() -> (potential!(μ, C, γ); concentration!(C, μ, dtD)), backend)
    ## arrays each kernel moves
    narr_pot  = 2  #sol
    #hint narr_pot  = ???
    narr_con  = 3  #sol
    #hint narr_con  = ???
    narr_step = 5                              # the minimum for our two-pass algorithm
    return [T_eff(narr_pot, C, t_pot), T_eff(narr_con, C, t_con), T_eff(narr_step, C, t_step)]
end

T_kp     = bench_ka(backend, FT, ns[end], γ, dt * D)
T_eff_ka = T_kp[end]                           # the whole step, kept for the final comparison
labels   = ["potential_ka!", "concentration_ka!", "whole step"]
for (label, T) in zip(labels, T_kp)
    @printf("%-18s %7.1f GB/s   %3.0f%% of T_peak\n", label, T, 100 * T / T_peak)
end

# The whole step now runs close to `T_peak`: the kernels move only the 5 arrays the
# algorithm needs, at nearly the speed of a memcopy. The stencil itself costs a
# little (reading neighbours, the boundary `min`/`max`), which is why we stay
# somewhat below the memcopy.
#
# ### Size matters, again
#
# The whole step for all the sizes of section 5, next to the memcopy:

function ka_sweep(backend, FT, ns, γ, dtD)
    T_step = Float64[]
    for n in ns
        C = KernelAbstractions.allocate(backend, FT, n, n)
        copyto!(C, initial_condition(FT, n))
        μ = KernelAbstractions.zeros(backend, FT, n, n)
        potential!     = potential_ka!(backend, 256, (n, n))
        concentration! = concentration_ka!(backend, 256, (n, n))
        t = time_it(() -> (potential!(μ, C, γ); concentration!(C, μ, dtD)), backend)
        push!(T_step, T_eff(5, C, t))
        @printf("n = %5d   Cahn-Hilliard step: %7.1f GB/s\n", n, T_step[end])
    end
    return T_step
end

T_step_ka = ka_sweep(backend, FT, ns, γ, dt * D)

#-

fig = Figure(size=(600, 400))
ax  = Axis(fig[1, 1]; xscale=log2, xticks=ns, xlabel="n  (arrays of n × n)",
           ylabel="T_eff [GB/s]", title="kernel programming, $(nameof(typeof(backend))), $FT")
scatterlines!(ax, ns, T_ka;        label="KA memcopy")
scatterlines!(ax, ns, T_step_ka;   label="Cahn-Hilliard step")
hlines!(ax, T_peak; color=:gray, linestyle=:dash, label="T_peak")
axislegend(ax; position=:rb)
fig

# As for the memcopy, small grids are dominated by the launch overhead. On large
# grids, the cost per cell no longer depends on the size: a 4× larger grid takes 4×
# longer per step. In grid units the time step does not depend on the resolution, so
# a larger grid simply means a larger domain at the same cost per cell.
#
# ### The payoff
#
# Let's run the Cahn-Hilliard solver on a 2048 × 2048 grid: 8× more cells along each
# direction than the CPU reference of section 7, so 64× more cells in total, for the
# same number of steps.
#
# *On a full GPU, try `n_big = 8192` (the size of our benchmarks): it takes about
# 40 s on an A100 or MI250X, but several minutes on a small MIG slice or a laptop,
# and plotting 67 million cells takes a while too.*

n_big = 2048
haskey(ENV, "CI") && (n_big = 256)             # CI only: small size, fast run  #src
C0_big = initial_condition(FT, n_big)
cahn_hilliard_ka(backend, C0_big, γ, dt * D, 10)    # warm-up: compiles for this size
t0 = time()
C_big = cahn_hilliard_ka(backend, C0_big, γ, dt * D, nt)
t_big = time() - t0
@printf("GPU, %d² cells: %.1f s,  %.3f ns per cell and step\n", n_big, t_big, t_big / nt / n_big^2 * 1e9)
@printf("CPU, %d² cells:   %.1f s,  %.3f ns per cell and step\n", n, t_cpu, t_cpu / nt / n^2 * 1e9)

#-

fig = Figure(size=(560, 500))
ax  = Axis(fig[1, 1]; aspect=DataAspect(), xlabel="x", ylabel="y",
           title=@sprintf("%d² cells, t = %.1f", n_big, nt * dt))
hm  = heatmap!(ax, C_big; colormap=:balance, colorrange=(-1, 1))
Colorbar(fig[1, 2], hm; label="C")
fig

# ### All implementations side by side
#
# The effective memory throughput of one time step, for every version we wrote:

T_eff_cpu = T_eff(5, C0, t_cpu / nt)           # the CPU reference of section 7
versions  = ["CPU loops\n(1 core, $(n)²)", "array\nprogramming", "kernel\nprogramming"]
T_all     = [T_eff_cpu, T_eff_ap, T_eff_ka]

fig = Figure(size=(600, 420))
ax  = Axis(fig[1, 1]; xticks=(1:3, versions), ylabel="T_eff [GB/s]",
           title="Cahn-Hilliard, one time step, $(nameof(typeof(backend))), $FT")
barplot!(ax, 1:3, T_all; bar_labels=:y, label_formatter=x -> @sprintf("%.0f", x))
hlines!(ax, T_peak; color=:gray, linestyle=:dash, label="T_peak")
axislegend(ax; position=:lt)
fig

#-

@printf("kernel programming is %.1f× faster than array programming\n", T_eff_ka / T_eff_ap)
@printf("and reaches %.0f%% of T_peak\n", 100 * T_eff_ka / T_peak)

# ### Share your results
#
# The line printed below extends the one of section 6 with the whole time step of
# array and kernel programming (`T_eff` in GB/s). Copy it into the results form.

println(join((device_name, nameof(typeof(backend)), FT, round(T_copyto[end]; digits=1),
              round(T_broadcast[end]; digits=1), round(T_memcopy_ka; digits=1),
              round(T_eff_ap; digits=1), round(T_eff_ka; digits=1)), "; "))

# ## Wrap-up and outlook
#
# ### What we have seen
#
# - Stencil-based PDE solvers are **memory bound**. We measure them with the
#   effective memory throughput `T_eff`, and compare it with `T_peak`, the memcopy
#   measured on our own device.
# - **Array programming** runs on the GPU as is, with broadcasting. It is quick to
#   write, but each statement is a kernel, and every intermediate array costs memory
#   traffic.
# - **Kernel programming** with KernelAbstractions: the kernel body is the loop body.
#   One kernel per pass keeps the intermediate values in registers, and gets close
#   to `T_peak`.
# - The same code runs on NVIDIA, AMD and Apple GPUs, and on the CPU.
#
# ### Challenge: even fewer arrays
#
# Our two-pass algorithm stores `μ` between the passes. But `μ` does not depend on
# its own history: we can **recompute it instead of storing it**. `∇²μ` needs `μ`
# at the four neighbours, so compute it there on the fly, from `C`, with a function
# such as
#
# ```julia
# Base.@propagate_inbounds function mu(C, ix, iy, nx, ny, γ)
#     c = C[ix, iy]
#     return c^3 - c - γ * lap(C, ix, iy, nx, ny)
# end
# ```
#
# A single kernel per time step then reads `C` and writes the new `C`: **2 arrays
# instead of 5**. Flops are (almost) free, remember?
#
# **Try it!** Some hints:
#
# - `C` can no longer be updated in place: the neighbours two cells away still need
#   the old values. Write into a second array `C2`, and swap the two after each
#   step: `C, C2 = C2, C`.
# - Check the result against `C_ref`, as before.
# - With our count of 5 arrays, `T_eff` can now exceed `T_peak`: you move less data
#   than the two-pass algorithm needs. Count the 2 arrays you actually move.
# - **No speed-up?** Then something other than memory limits the kernel. Here, it is
#   the `min` and `max` of the mirror boundaries. In `mu(C, max(ix-1, 1), iy, ...)`,
#   the compiler cannot see that many neighbour loads are the same, so it loads,
#   and computes addresses for, about 25 values instead of 13. A fast path for the
#   interior cells with plain offsets (`C[ix-1, iy]`, ...), with `min` and `max`
#   only near the boundary, lets it share them. On an AMD MI250X at 8192², we
#   measured 2.26 ms per step for the two kernels, 2.20 ms for the simple fused
#   kernel, and 1.52 ms for the fused kernel with an interior fast path.
#
# A reference solution is in [`extras/fused_step.jl`](../extras/fused_step.jl).
#
# ### Going further
#
# - **More GPUs:** [ParallelStencil.jl](https://github.com/omlins/ParallelStencil.jl)
#   and [ImplicitGlobalGrid.jl](https://github.com/eth-cscs/ImplicitGlobalGrid.jl)
#   write stencil kernels in a notation close to the maths, and run them on many
#   GPUs, with the halo exchange between them handled for you.
# - **Higher-level finite differences:** [Chmy.jl](https://github.com/PTsolvers/Chmy.jl)
#   builds dimension-agnostic stencils on staggered grids, on top of
#   KernelAbstractions.
# - **More kinds of devices:** [Reactant.jl](https://github.com/EnzymeAD/Reactant.jl)
#   compiles Julia code through MLIR and XLA, for CPUs, GPUs and TPUs.
# - **A full course:** [Solving PDEs in parallel on GPUs with Julia](https://pde-on-gpu.vaw.ethz.ch),
#   ETH Zurich.
# - **The documentation:** [KernelAbstractions.jl](https://juliagpu.github.io/KernelAbstractions.jl/stable/).
