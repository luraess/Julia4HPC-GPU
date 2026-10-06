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

Now choose the device to run on. The CPU is active by default: it is only there
to check that the code runs, the performance sections are meant for a GPU.

To use a GPU, comment out the two CPU lines and uncomment the three lines of your GPU
(select them and press `Ctrl + /`). On Arctic, that is the NVIDIA block on the
`ar_mig` and `ar_a100` partitions, and the AMD block on `ar_mi210`.

`FT` is the floating-point type we compute in. Apple GPUs do not support `Float64`.

````julia
backend = CPU();  FT = Float64                      # CPU: only to check that the code runs
device_name = Sys.cpu_info()[1].model

# using CUDA                                       # NVIDIA
# backend = CUDABackend();  FT = Float64
# device_name = CUDA.name(CUDA.device())           # which GPU (or MIG slice) you got

# using AMDGPU                                     # AMD
# backend = ROCBackend();  FT = Float64
# device_name = AMDGPU.HIP.name(AMDGPU.device())   # which GPU you got

# using Metal                                      # Apple laptop
# backend = MetalBackend();  FT = Float32
# device_name = string(Metal.device().name)

println("backend = ", nameof(typeof(backend)), ",  FT = ", FT, ",  device = ", device_name)
````

## 1. Why GPUs

Solving PDEs gets expensive fast. The equation we will solve, Cahn-Hilliard, is
fourth order in space: with an explicit scheme the time step shrinks as `dt ∝ dx⁴`.
Halving the grid spacing in 2D means 4× more cells *and* 16× more time steps, so
**64× more work**.

Since the mid-2000s, processors have stopped getting faster per core. Performance
now comes from parallelism: more cores, wider vector units. GPUs take this
furthest, with thousands of lightweight threads. More important for us, they also
have **much higher memory bandwidth** than CPUs:

Here is the hardware of the Arctic nodes we run on:

| device | FP64 peak [TFLOP/s] | memory bandwidth [GB/s] | balance [flop per number] |
|:-------|-------:|-------:|-----:|
| AMD EPYC 7543 (CPU, one socket, 32 cores) | 1.4 | 205 | ~56 |
| NVIDIA A100 SXM 80 GB | 9.7 | 2039 | ~38 |
| AMD Instinct MI210 | 22.6 | 1638 | ~110 |

*Vendor peak values. An A100 node has two EPYC 7543 sockets and eight A100s. The
balance is FLOP/s ÷ (bytes/s) × 8 bytes: how many floating-point operations a
device can do in the time it takes to load one `Float64` from memory.*

Two things to take from this table:
- A GPU has about **10× the memory bandwidth** of a CPU socket.
- Every device can do **40–110 flops per number it loads**. Unless a code does
  that much arithmetic per number, it waits for memory, not for compute
  (section 3).

For such memory-bound codes, the speedup of a GPU over a CPU is about the ratio
of their memory bandwidths.

On Arctic, the `ar_mig` partition gives you a *slice* of an A100 (MIG). A slice
gets a share of the memory bandwidth: roughly 1/8 for `1g.10gb`, 1/4 for `2g.20gb`
and 1/2 for `3g.40gb`. We will measure what you actually get in section 5.

## 2. The problem: Cahn-Hilliard in 2D

The [Cahn-Hilliard equation](https://en.wikipedia.org/wiki/Cahn%E2%80%93Hilliard_equation)
describes how a binary mixture, think oil and water, separates into two phases:

```math
\frac{\partial C}{\partial t} = D \nabla^2 \mu , \qquad \mu = C^3 - C - \gamma \nabla^2 C
```

- `C` is the concentration, between -1 and 1 (the two pure phases).
- `μ` is the chemical potential, `D` the mobility, and `γ` sets the width of the
  interface between the phases.
- The boundaries are closed (no flux), so the amount of each phase is conserved:
  **`mean(C)` stays constant**.
- The free energy **`F` decreases** over time.

These last two properties will be our correctness checks.

Starting from small random noise, the phases separate and then coarsen. The
conserved mean `C̄ = mean(C)` selects the pattern: interwoven bands for `C̄ = 0`
(left), droplets for `C̄ = 0.4` (right):

[▶ `C̄ = 0`](../assets/CahnHilliard2D_C0.mp4) · [▶ `C̄ = 0.4`](../assets/CahnHilliard2D_C04.mp4)

What matters for performance is that each time step makes **two passes** over the
grid:
1. compute `μ` from `C`, which needs the neighbours of `C` for `∇²C`;
2. update `C` from `μ`, which needs the neighbours of `μ` for `∇²μ`.

Since pass 2 needs `μ` at the neighbouring cells, `μ` has to be stored in an array
between the two passes. Each time step therefore reads `C` and writes `μ`, then
reads `μ` and `C` and writes `C`: **5 array accesses** per time step.

## 3. What limits performance

A computation is limited either by how fast the device does arithmetic
(**compute bound**), or by how fast it moves data to and from memory
(**memory bound**).

Let's count for pass 1, `μ = C³ - C - γ∇²C`, at one grid cell, with `c = C[ix, iy]`:

```julia
μ[ix, iy] = c*c*c - c - γ * (C[ix-1, iy] + C[ix+1, iy] + C[ix, iy-1] + C[ix, iy+1] - 4c)
```

- **Memory:** read `C`, write `μ`: 2 numbers, 16 bytes. The neighbours of `C`
  were already loaded for the neighbouring cells, so they come from cache.
- **Arithmetic:** about 10 floating-point operations.

That is ~5 flops per number moved, where the devices above could do 40–110. The
GPU spends most of its time waiting for memory: **flops are (almost) free, bytes
are expensive**.

So FLOP/s is the wrong metric for this kind of code. What we need to measure is
how fast we move memory.

## 4. Memory bound: memcopy is the ceiling

For memory-bound codes we measure the **effective memory throughput**

```math
T_\mathrm{eff} = \frac{n_\mathrm{arrays} \cdot n_x \cdot n_y \cdot \mathrm{sizeof(FT)}}{10^9 \cdot t} \quad [\mathrm{GB/s}]
```

where `sizeof(FT)` is the size of one number in bytes (8 for `Float64`), `t` is
the time of one call (or one time step) in seconds, and `n_arrays` counts the
arrays that **must** be read or written, assuming neighbours come from cache:

| operation | arrays moved |
|:----------|:-------------|
| memcopy `A = B` | 2: read `B`, write `A` |
| Cahn-Hilliard, one time step | 5: read `C`, write `μ`, read `μ`, read `C`, write `C` |

A memcopy does nothing but move data, so no code moving the same amount of data
can be faster. **The memcopy throughput measured on your GPU, `T_peak`, is our
ceiling**, and we will report every implementation as a fraction of it.

Why measure it rather than take the vendor's number? Vendor peaks are not reached
in practice (80–90% of them is typical), and on a MIG slice you only get part of
the GPU. Measuring tells us what *your* device can actually do.

## 5. Measuring memcopy without KernelAbstractions

KernelAbstractions gives us a portable way to allocate arrays on the device:
`KernelAbstractions.allocate(backend, FT, nx, ny)` (uninitialised) and
`KernelAbstractions.zeros(backend, FT, nx, ny)`. On NVIDIA they return a
`CuArray`, on AMD a `ROCArray`, on the CPU a plain `Array`:

````julia
A_small = KernelAbstractions.zeros(backend, FT, 4, 4)
typeof(A_small)
````

In this section we only use KernelAbstractions to get the arrays. The copies are
done without writing any kernel ourselves: with `copyto!(A, B)`, the vendor's
tuned copy, and with broadcasting, `A .= B`, for which Julia generates a GPU
kernel for us.

### GPU operations are asynchronous

Launching work on a GPU returns immediately, before the work is done. To time it,
we must wait for the GPU to finish with `KernelAbstractions.synchronize(backend)`:

````julia
n = 8192
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
````

On the CPU both numbers are about the same: there, everything is synchronous.

### Timing and T_eff

Two small helpers. `time_it` runs `f` once to compile and warm up, then times
`nrep` calls and keeps the best of `ntrial` trials, to filter out noise from
other users on the machine:

````julia
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
````

`T_eff` follows the formula of section 4, in GB/s, for arrays the size of `A`:

````julia
T_eff(narrays, A, t) = narrays * length(A) * sizeof(eltype(A)) / t / 1e9
````

Back to our 8192 × 8192 copy, which moves 2 arrays:

````julia
t = time_it(() -> copyto!(A, B), backend)
@printf("copyto!: t = %.3f ms,  T_eff = %.1f GB/s\n", t * 1e3, T_eff(2, A, t))
````

### Size matters

Let's repeat this for several array sizes. Note that the loop lives inside a
function: in Julia, code in functions is compiled and fast, and the variables
inside do not clash with the ones of the notebook.

````julia
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
T_copyto, T_broadcast = memcopy_sweep(backend, FT, ns)
````

````julia
fig = Figure(size=(600, 400))
ax  = Axis(fig[1, 1]; xscale=log2, xticks=ns, xlabel="n  (arrays of n × n)",
           ylabel="T_eff [GB/s]", title="memcopy, $(nameof(typeof(backend))), $FT")
scatterlines!(ax, ns, T_copyto;    label="copyto!")
scatterlines!(ax, ns, T_broadcast; label="A .= B")
axislegend(ax; position=:lt)
fig
````

Small arrays do not measure the memory bandwidth: the launch overhead dominates,
or the arrays fit in the GPU's cache. Only the largest sizes tell us what the
memory can do. As a first estimate of our ceiling, we take the vendor copy at the
largest size:

````julia
T_peak = T_copyto[end]
@printf("T_peak = %.1f GB/s\n", T_peak)
````

How does your `T_peak` compare to the vendor number in section 1, or to your
share of it on a MIG slice?

## 6. A first kernel: memcopy with KernelAbstractions

Now we write the copy ourselves. On the CPU, a 2D copy is a double loop:

```julia
for iy in 1:ny, ix in 1:nx
    A[ix, iy] = B[ix, iy]
end
```

A GPU kernel keeps only the **body of the loop**: it describes what happens at one
`(ix, iy)`. The GPU then runs it for all `(ix, iy)` at once, one *work-item* (a
GPU thread) each. With KernelAbstractions:

- `@kernel` turns a function into a kernel. A kernel returns nothing, it writes
  into its arguments.
- `@index(Global, NTuple)` gives the work-item its `(ix, iy)`.
- `inbounds = true` switches off bounds checking inside the kernel. We will make
  sure that every `(ix, iy)` is inside the arrays.

````julia
@kernel inbounds = true function memcopy_ka!(A, B)
    ix, iy = @index(Global, NTuple)
    A[ix, iy] = B[ix, iy]
end
````

Running a kernel takes two steps. First we *instantiate* it for our backend, then
we *launch* it, giving the `ndrange`: the range of `(ix, iy)` to run over. Here
we want one work-item per array element:

````julia
memcopy_dyn! = memcopy_ka!(backend)          # 1. instantiate for our backend
memcopy_dyn!(A, B; ndrange = size(A))        # 2. launch over all (ix, iy)
KernelAbstractions.synchronize(backend)      # 3. wait for the GPU to finish

@assert Array(A) == Array(B)                 # copy back to the CPU and compare
println("memcopy_ka! works")
````

### Static or dynamic launch

Above, the sizes were only given at launch: the kernel is *dynamic*. We can also
fix them when instantiating: the *workgroup size*, i.e. how many work-items are
grouped together on the GPU (256 is a good default on all GPUs), and the `ndrange`.
The compiler then knows them in advance:

````julia
memcopy_static! = memcopy_ka!(backend, 256, size(A))   # workgroup size and ndrange fixed
memcopy_static!(A, B)                                  # no ndrange needed at launch

t_dyn    = time_it(() -> memcopy_dyn!(A, B; ndrange = size(A)), backend)
t_static = time_it(() -> memcopy_static!(A, B), backend)

@printf("dynamic: %7.1f GB/s\n", T_eff(2, A, t_dyn))
@printf("static:  %7.1f GB/s\n", T_eff(2, A, t_static))
@printf("copyto!: %7.1f GB/s\n", T_copyto[end])
````

The static kernel is usually faster: knowing the sizes lets the compiler simplify
how each work-item computes its `(ix, iy)`. Section 10 looks at what `@index`
actually does. From now on, we always use static kernels.

### How close to the ceiling?

Let's sweep the sizes again with the static kernel. **Exercise:** how many arrays
does a memcopy move?

````julia
function memcopy_ka_sweep(backend, FT, ns)
    T_ka = Float64[]
    for n in ns
        A = KernelAbstractions.zeros(backend, FT, n, n)
        B = KernelAbstractions.allocate(backend, FT, n, n)
        copyto!(B, rand(FT, n, n))
        memcopy! = memcopy_ka!(backend, 256, (n, n))
        t = time_it(() -> memcopy!(A, B), backend)
        push!(T_ka, T_eff(2, A, t))
        @printf("n = %5d   KA memcopy: %7.1f GB/s\n", n, T_ka[end])
    end
    return T_ka
end

T_ka = memcopy_ka_sweep(backend, FT, ns)
````

````julia
T_memcopy_ka = T_ka[end]
@printf("KA memcopy: %.1f GB/s = %.0f%% of copyto!\n", T_memcopy_ka, 100 * T_memcopy_ka / T_copyto[end])
````

A few lines of portable Julia get close to the vendor's tuned copy, and on some
GPUs (AMD MI200 series, for example) they even beat it. The vendor copy is not
always the fastest way to copy, so **our ceiling is the fastest memcopy we
measured**:

````julia
T_peak = max(T_copyto[end], T_broadcast[end], T_memcopy_ka)
@printf("T_peak = %.1f GB/s\n", T_peak)
````

````julia
fig = Figure(size=(600, 400))
ax  = Axis(fig[1, 1]; xscale=log2, xticks=ns, xlabel="n  (arrays of n × n)",
           ylabel="T_eff [GB/s]", title="memcopy, $(nameof(typeof(backend))), $FT")
scatterlines!(ax, ns, T_copyto;    label="copyto!")
scatterlines!(ax, ns, T_broadcast; label="A .= B")
scatterlines!(ax, ns, T_ka;        label="KA kernel")
hlines!(ax, T_peak; color=:gray, linestyle=:dash, label="T_peak")
axislegend(ax; position=:lt)
fig
````

`T_peak` is the baseline for the Cahn-Hilliard kernels: they cannot beat it, and
we will see how close they get.

*On an Apple GPU, the KA kernel stays well below `copyto!`. Section 10 explains
why, and how to fix it.*

### Share your results

Copy the line printed below into the results form (link given during the
workshop). It holds your device and the three memcopy throughputs, in GB/s:
`copyto!`, `A .= B` and the KA kernel. We will compare the GPUs of the whole room.

````julia
println(join((device_name, nameof(typeof(backend)), FT, round(T_copyto[end]; digits=1),
              round(T_broadcast[end]; digits=1), round(T_memcopy_ka; digits=1)), "; "))
````

## 7. Cahn-Hilliard: discretisation and CPU reference

## 8. Cahn-Hilliard with array programming

## 9. Performance of array programming

## 10. Kernel programming with neighbours

## 11. Performance of kernel programming

## Outlook

