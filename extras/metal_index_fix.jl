# Temporary fix for slow indexing on Apple GPUs. The Metal block of the notebook and
# of extras/fused_step.jl loads it right after `using Metal`.
#
# The problem (https://github.com/JuliaGPU/Metal.jl/issues/910): the GPU gives each
# work-item two numbers, its workgroup and its place inside the workgroup. Metal.jl
# turns them into `(ix, iy)` with integer divisions, in 64-bit integers, which Apple
# GPUs emulate in software. Kernels using `@index` then run at about 20% of the
# memory bandwidth. Broadcasts hit the same problem when the first dimension of the
# result is not a power of two, such as `qx[2:nx, :] .= ...` in section 8.
#
# The fix: the same arithmetic in 32-bit integers, when the launch has fewer than
# 2^32 work-items. It replaces methods inside Metal.jl, so it depends on internals of
# Metal.jl and KernelAbstractions: tested with Metal.jl 1.11 and KernelAbstractions
# 0.9. Delete this file once Metal.jl does this itself.
module MetalIndexFix

using Metal
import KernelAbstractions as KA

# only apply the fix to the versions it was tested with
const APPLY = pkgversion(Metal) < v"2" && pkgversion(KA) < v"0.10"
APPLY || @warn "extras/metal_index_fix.jl was not applied: it was tested with Metal.jl 1.x and KernelAbstractions 0.9 only"

# 0-based linear index -> 0-based indices along each dimension, in 32 bits.
# `dims` are the sizes along each dimension; column-major, like Julia arrays.
@inline ind2sub0(::Tuple{UInt32}, i::UInt32) = (i,)
@inline function ind2sub0(dims::Tuple{UInt32, UInt32, Vararg{UInt32}}, i::UInt32)
    q = div(i, dims[1])
    return (i - q * dims[1], ind2sub0(Base.tail(dims), q)...)
end

# sizes of a CartesianIndices, in 32 bits
@inline size32(CI::CartesianIndices) = map(r -> length(r) % UInt32, CI.indices)

# ---- 1. KernelAbstractions kernels: @index(Global, ...) ----

# The global index of a work-item, from its workgroup number `g` and its number `l`
# inside the workgroup (both 1-based): the same as KA.expand, in 32 bits.
@inline function expand32(iterspace, g::UInt32, l::UInt32)
    B  = size32(KA.blocks(iterspace))          # workgroups along each dimension
    W  = size32(KA.workitems(iterspace))       # workgroup size along each dimension
    gI = ind2sub0(B, g - UInt32(1))
    lI = ind2sub0(W, l - UInt32(1))
    return CartesianIndex(map((gi, w, li) -> Int(gi * w + li + UInt32(1)), gI, W, lI))
end

# 32 bits are enough if the launch has fewer than 2^32 work-items. For static
# kernels, this is known at compile time and the check disappears.
@inline fits32(iterspace) =
    length(KA.blocks(iterspace)) * length(KA.workitems(iterspace)) <= typemax(UInt32)

if APPLY
    Metal.@device_override @inline function KA.__index_Global_Cartesian(ctx)
        is = KA.__iterspace(ctx)
        g, l = threadgroup_position_in_grid().x, thread_position_in_threadgroup().x
        return fits32(is) ? expand32(is, g, l) : @inbounds KA.expand(is, g, l)
    end

    Metal.@device_override @inline function KA.__index_Global_Linear(ctx)
        is = KA.__iterspace(ctx)
        g, l = threadgroup_position_in_grid().x, thread_position_in_threadgroup().x
        if fits32(is)
            I  = expand32(is, g, l)
            nd = size32(KA.__ndrange(ctx))
            lin, stride = UInt32(0), UInt32(1)         # column-major linear index
            for d in 1:length(nd)
                lin += (I.I[d] % UInt32 - UInt32(1)) * stride
                stride *= nd[d]
            end
            return Int(lin + UInt32(1))
        else
            I = @inbounds KA.expand(is, g, l)
            return @inbounds LinearIndices(KA.__ndrange(ctx))[I]
        end
    end

    Metal.@device_override @inline function KA.__validindex(ctx)
        if KA.__dynamic_checkbounds(ctx)
            is = KA.__iterspace(ctx)
            g, l = threadgroup_position_in_grid().x, thread_position_in_threadgroup().x
            I = fits32(is) ? expand32(is, g, l) : @inbounds KA.expand(is, g, l)
            return I in KA.__ndrange(ctx)
        else
            return true
        end
    end
end

# ---- 2. Broadcasts ----

# After 10 broadcasts with the same shape, Metal.jl switches to a kernel that turns a
# linear index into a CartesianIndex through `StaticCartesianIndices`. Same fix.
if APPLY && isdefined(Metal, :StaticCartesianIndices)
    @inline function Base.getindex(::Metal.StaticCartesianIndices{N, I}, i::Int) where {N, I}
        if prod(map(length, I)) <= typemax(UInt32)    # known at compile time
            dims = map(r -> length(r) % UInt32, I)
            sub  = ind2sub0(dims, (i - 1) % UInt32)
            return CartesianIndex(map((s, r) -> Int(s) + first(r), sub, I))
        else
            return @inbounds CartesianIndices(I)[i]
        end
    end
end

end # module
