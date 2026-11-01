## Whisk method-table overrides
##
## Mirrors CUDA's `@device_override` pattern. Overlays go on the
## *lowest-level* helpers whose bodies fall through to a host-VM
## ccall we can't lower. Most of the time the right intercept is
## actually below the Julia method — we provide the C symbol as
## a wasm-side runtime function instead (see `runtime/gc.jl` for
## `jl_genericmemory_copyto`). Empty for now.

using Base.Experimental: @overlay, @MethodTable
@MethodTable method_table

# ── Threads.Atomic ops — wasm builds are single-threaded, so the
# host's lock-based / LLVM-atomicrmw paths degrade to plain field
# updates. Without these overlays every `atomic_*` call on a global
# counter (e.g. Observables' OBSID_COUNTER) becomes an unresolvable
# dynamic site through host-runtime atomics.
@overlay method_table function Base.Threads.atomic_add!(x::Base.Threads.Atomic{T}, v::T) where {T}
    old = x[]
    x[] = old + v
    return old
end
@overlay method_table function Base.Threads.atomic_sub!(x::Base.Threads.Atomic{T}, v::T) where {T}
    old = x[]
    x[] = old - v
    return old
end
