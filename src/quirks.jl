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
