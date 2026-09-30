# Julia4HPC-GPU

<a href="https://julia4hpc.sciencesconf.org/program/graphic/date/2026-10-14"><img src="assets/julia4hpc_logo.png" alt="Julia4HPC" width="200" align="right"></a>

GPU computing in Julia for HPC: the GPU part of [**Julia4HPC**, Formation au Langage Julia pour le Calcul Haute Performance](https://julia4hpc.sciencesconf.org/program/graphic/date/2026-10-14), Fréjus (France), 12–16 October 2026.

**When:** Wednesday 14 October 2026, 9:00–10:20 and 10:40–12:00, room Mimosa

**Instructor:** [Ludovic Räss](https://github.com/luraess)

<br clear="right">

We take one equation, Cahn-Hilliard in 2D, from a plain CPU loop to a GPU kernel running close to the memory bandwidth of the device, measuring at every step. The code uses [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl) throughout, so it runs unchanged on NVIDIA, AMD and Apple GPUs, and on the CPU.

## Getting started

### On Arctic (MesoNET, CRIANN)

<!-- TODO: JupyterHub URL, login steps and the name of the Julia kernel -->

The workshop runs on [Arctic](https://services.criann.fr/services/hpc/cluster-austral/guide/#service-mesonet-arctic), the MesoNET training partition of the Austral cluster at CRIANN. Start a JupyterHub session on one of these partitions:

| partition | GPU | backend |
|---|---|---|
| `ar_mig` | slice of an NVIDIA A100 (MIG) | CUDA |
| `ar_a100` | NVIDIA A100 | CUDA |
| `ar_mi210` | AMD MI210 (1 h session limit) | AMDGPU |

Then clone this repository and open [`notebooks/gpu_workshop.ipynb`](notebooks/gpu_workshop.ipynb).

### On your own machine

```bash
git clone https://github.com/luraess/Julia4HPC-GPU.git
cd Julia4HPC-GPU
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

On an Apple laptop, also run `julia --project=. -e 'using Pkg; Pkg.add("Metal")'`. Then open [`notebooks/gpu_workshop.ipynb`](notebooks/gpu_workshop.ipynb) in Jupyter or VS Code.

## Material

| file | content |
|---|---|
| [`notebooks/gpu_workshop.ipynb`](notebooks/gpu_workshop.ipynb) | the workshop notebook, with blanks to fill in |
| [`solutions/gpu_workshop_solution.ipynb`](solutions/gpu_workshop_solution.ipynb) | the same notebook, completed |
| [`docs/gpu_workshop.md`](docs/gpu_workshop.md) | the completed notebook as a page to read |

## For authors

All three files above are generated from a single [Literate.jl](https://github.com/fredrikekre/Literate.jl) source, [`src/gpu_workshop.jl`](src/gpu_workshop.jl). Never edit them by hand: edit the source, then regenerate them from the repo root:

```bash
julia --project=deploy -e 'using Pkg; Pkg.instantiate(); include("deploy/deploy.jl")'
```

Mark exercise lines in the source:

```julia
    ix, iy = @index(Global, NTuple)  #sol     # answer: only in the solution
    #hint ix, iy = ???                        # blank: only in the exercise notebook
```

The source is the solution and runs as a plain Julia script. CI checks that the generated files match the source and runs the solution on the CPU backend.
