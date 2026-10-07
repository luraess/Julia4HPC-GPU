# Generate the participant-facing files from the single Literate source in src/:
#
#   src/gpu_workshop.jl -> notebooks/gpu_workshop.ipynb           exercise, with blanks
#                       -> solutions/gpu_workshop_solution.ipynb  solution
#                       -> docs/gpu_workshop.md                   solution, for reading on GitHub
#
# Run from the repo root:
#
#   julia --project=deploy -e 'using Pkg; Pkg.instantiate(); include("deploy/deploy.jl")'
#
# Never edit the generated files by hand: edit src/ and re-run this script.
# CI re-runs it and fails if the committed files differ.
using Literate

const ROOT = normpath(joinpath(@__DIR__, ".."))
const SRC  = joinpath(ROOT, "src", "gpu_workshop.jl")

# Jupyter kernel the notebooks open with. Left to itself, Literate writes the Julia
# version of whoever runs this script, so the output would depend on the machine.
# Arctic runs Julia 1.13; the workshop setup installs its kernel with
# `IJulia.installkernel("Julia", ...)`, which names it "julia-1.13".
const KERNEL_NAME    = "julia-1.13"
const KERNEL_DISPLAY = "Julia 1.13"

# Exercise / solution markers, one line at a time:
#
#   <code>  #sol    kept in the solution (marker stripped), removed from the exercise
#   #hint <code>    uncommented in the exercise, removed from the solution
#
# The source is the solution and runs as-is, since `#hint` lines are plain comments.
is_sol(line)  = occursin(r"#sol\s*$", line)
is_hint(line) = occursin(r"^\s*#hint\b", line)

function exercise(content)
    lines = String[]
    for line in split(content, '\n')
        if is_sol(line)
            continue                                              # drop the answer
        elseif is_hint(line)
            push!(lines, replace(line, r"^(\s*)#hint ?" => s"\1")) # show the blank
        else
            push!(lines, line)
        end
    end
    return join(lines, '\n')
end

function solution(content)
    lines = String[]
    for line in split(content, '\n')
        if is_hint(line)
            continue                                              # drop the blank
        elseif is_sol(line)
            push!(lines, replace(line, r"\s*#sol\s*$" => ""))     # keep the answer
        else
            push!(lines, line)
        end
    end
    return join(lines, '\n')
end

function pin_kernel(nb)
    nb["metadata"]["kernelspec"] = Dict("language"     => "julia",
                                        "name"         => KERNEL_NAME,
                                        "display_name" => KERNEL_DISPLAY)
    nb["metadata"]["language_info"] = Dict("name"           => "julia",
                                           "file_extension" => ".jl",
                                           "mimetype"       => "application/julia")
    return nb
end

name = splitext(basename(SRC))[1]

Literate.notebook(SRC, joinpath(ROOT, "notebooks"); name,
                  preprocess=exercise, postprocess=pin_kernel, execute=false, credit=false)

Literate.notebook(SRC, joinpath(ROOT, "solutions"); name=name * "_solution",
                  preprocess=solution, postprocess=pin_kernel, execute=false, credit=false)

Literate.markdown(SRC, joinpath(ROOT, "docs"); name,
                  preprocess=solution, flavor=Literate.CommonMarkFlavor(), execute=false, credit=false)
