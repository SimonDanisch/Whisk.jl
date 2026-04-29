## GPUCompiler integration

"""
WasmCompilerTarget — emit code for `wasm32-unknown-unknown`.

Subtypes `AbstractCompilerTarget`. Provides the LLVM triple, datalayout,
and TargetMachine that GPUCompiler needs for code generation. The wasm
backend is registered at `Whisk.__init__` time.
"""
Base.@kwdef struct WasmCompilerTarget <: AbstractCompilerTarget
    # Reserved for future feature flags (e.g. SIMD, threads, multi-memory).
    # Empty for now — the default wasm32-unknown-unknown is enough for M1.
    features::String = ""
end

GPUCompiler.llvm_triple(::WasmCompilerTarget) = "wasm64-unknown-unknown"

function GPUCompiler.llvm_machine(target::WasmCompilerTarget)
    triple = GPUCompiler.llvm_triple(target)
    t      = LLVM.Target(triple = triple)
    return LLVM.TargetMachine(t, triple, target.features)
end

# Wasm's default datalayout marks addrspace 10/20 as externref/funcref
# (`p10:8:8 p20:8:8 ni:1:10:20`). GPUCompiler's `julia_datalayout`
# adds `ni:10:11:12:13` on top. Override to drop the `ni:10:…` —
# combined with `uses_julia_runtime = false` (default), Julia's
# frontend emits regular addrspace-0 pointers and the standard wasm
# backend lowering works.
GPUCompiler.julia_datalayout(target::WasmCompilerTarget) =
    GPUCompiler.llvm_datalayout(target)

"""
WasmCompilerParams — per-job knobs.

For now there's nothing per-job to vary, but the type exists so we have
a place to add e.g. allocator choice, debug, exception model.
"""
Base.@kwdef struct WasmCompilerParams <: AbstractCompilerParams
end

const WasmCompilerConfig = CompilerConfig{WasmCompilerTarget, WasmCompilerParams}
const WasmCompilerJob    = CompilerJob{WasmCompilerTarget, WasmCompilerParams}

# `runtime_module` returns the module that holds Whisk-specific runtime
# stubs (allocator, exceptions, intrinsics) — GPUCompiler will resolve
# unresolved symbols against it. We just point it at the top-level
# `Whisk` module; the runtime files are included from there.
GPUCompiler.runtime_module(::WasmCompilerJob) = Whisk

# `method_table` is the overlay table for Whisk-specific stdlib overrides
# (the equivalent of CUDA's `@device_override`). Created in quirks.jl.
GPUCompiler.method_table(::WasmCompilerJob) = method_table

# Wasm has no concept of GPU-style kernel state; pass-by-value is fine.
# We can switch this off later if we want a `KernelState`-style global.
GPUCompiler.kernel_state_type(::WasmCompilerJob) = Nothing

# Cache slug for the disk-cached compiled-runtime image. Combines target
# triple + features so jobs differing in those don't share a cache.
GPUCompiler.runtime_slug(job::WasmCompilerJob) =
    "wasm64-$(job.config.target.features)"

# `finish_module!` runs AFTER IR generation but BEFORE optimize!, so
# we can rewrite `julia.gc_alloc_obj` calls here — pre-empting
# GPUCompiler's `lower_gc_frame!` pass which would otherwise call into
# a host-typed `gc_pool_alloc(i64)`. wasm32 needs an i32 size.
function GPUCompiler.finish_module!(job::WasmCompilerJob, mod::LLVM.Module, entry::LLVM.Function)
    job.config.toplevel || return entry

    emit_wasm_runtime!(mod)                    # allocator, Memory ctor, GC
    lower_gc_alloc_wasm!(mod)                  # julia.gc_alloc_obj → whisk_gc_alloc_typed
    lower_alloc_genericmemory!(mod)            # Memory<T> alloc with layout descriptor
    lower_gc_frame_intrinsics!(mod)            # 1.11+ GC frames → shadow stack
    bypass_argument_error_checks!(mod)         # bypass mistruncated bounds checks
    return entry
end

function GPUCompiler.finish_ir!(job::WasmCompilerJob, mod::LLVM.Module, entry::LLVM.Function)
    # Post-optimize. By this point Julia's `LateLowerGCPass`/
    # `FinalLowerGCPass` have produced real GC frames (allocas +
    # slot stores around `ijl_gc_small_alloc` calls) and the
    # `@julia.call2` wrappers have been resolved into `ijl_invoke`
    # calls with literal `inttoptr` operands. We finish wiring
    # things up in this order:
    #   1. `lower_julia_call2!` / `lower_invokes!` — resolve
    #      dispatch.
    #   2. `rewrite_julia_pgcstack_for_wasm!` — replace x86 inline
    #      asm reading TLS with our `_whisk_pgcstack` global.
    #   3. `stub_julia_safepoint_polls!` — erase volatile poll
    #      loads + leftover `julia.safepoint` calls.
    #   4. `emit_late_julia_runtime_stubs!` — provide bodies for
    #      `ijl_gc_small_alloc` / trap stubs for `ijl_error` etc.
    #   5. `lower_gc_alloc_wasm!` / `lower_gc_frame_intrinsics!` —
    #      catch any leftover gc_alloc_obj / new_gc_frame nodes
    #      from older sub-paths.
    #   6. Host-pointer rewrites + dispatch + trap reports.
    job.config.toplevel || return entry
    lower_julia_call2!(mod)
    lower_invokes!(mod, job.config.target, job.config.params)
    rewrite_host_pointer_constants!(mod)
    lower_dynamic_dispatch!(mod)
    lower_trap_reports!(mod)
    return entry
end

# Things we tell GPUCompiler's IR validator to *not* reject as
# "unsupported call". They become wasm-imported functions or are
# already lowered away by `finish_module!`. This is the same hook CUDA /
# Metal / Abacus use for runtime-resolved symbols.
const _whisk_intrinsics = Set([
    "whisk_gc_alloc",                    # our allocator (wasm-import for M2)
    "whisk_gc_collect",                  # mark+sweep, callable from Julia helpers
    "whisk_pin",                         # JS-roots pin, wasm export
    "whisk_unpin",                       # JS-roots unpin, wasm export
    "jl_alloc_genericmemory_unchecked",  # 1.11+ Memory{T} alloc — wasm import
    "jl_argument_error",                 # legacy error throw — wasm import
    "js_console",                        # console.log/warn/error import
    "gpu_report_exception",              # GPUCompiler error stub
    "gpu_report_oom",                    # OOM reporter
    "gpu_signal_exception",
    "memcpy", "memmove", "memset",
])

GPUCompiler.isintrinsic(::WasmCompilerJob, fn::String) = fn in _whisk_intrinsics
