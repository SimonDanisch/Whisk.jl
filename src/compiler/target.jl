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

GPUCompiler.llvm_triple(::WasmCompilerTarget) = "wasm32-unknown-unknown"

function GPUCompiler.llvm_machine(target::WasmCompilerTarget)
    triple = GPUCompiler.llvm_triple(target)
    t      = LLVM.Target(triple = triple)
    return LLVM.TargetMachine(t, triple, target.features)
end

# llvm_datalayout falls back to TargetMachine's data layout, which is
# what we want for wasm32 — no override needed.

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
    "wasm32-$(job.config.target.features)"

# `finish_module!` runs AFTER IR generation but BEFORE optimize!, so
# we can rewrite `julia.gc_alloc_obj` calls here — pre-empting
# GPUCompiler's `lower_gc_frame!` pass which would otherwise call into
# a host-typed `gc_pool_alloc(i64)`. wasm32 needs an i32 size.
function GPUCompiler.finish_module!(job::WasmCompilerJob, mod::LLVM.Module, entry::LLVM.Function)
    emit_wasm_runtime!(mod)             # in-wasm allocator + Memory ctor
    lower_gc_alloc_wasm32!(mod)         # julia.gc_alloc_obj → whisk_gc_alloc
    lower_gc_frame_intrinsics!(mod)     # erase 1.11+ GC frame intrinsics
    bypass_argument_error_checks!(mod)  # bypass mistruncated bounds checks
    return entry
end

# Things we tell GPUCompiler's IR validator to *not* reject as
# "unsupported call". They become wasm-imported functions or are
# already lowered away by `finish_module!`. This is the same hook CUDA /
# Metal / Abacus use for runtime-resolved symbols.
const _whisk_intrinsics = Set([
    "whisk_gc_alloc",                    # our allocator (wasm-import for M2)
    "jl_alloc_genericmemory_unchecked",  # 1.11+ Memory{T} alloc — wasm import
    "jl_argument_error",                 # error throw — wasm import / trap
    "gpu_report_exception",              # GPUCompiler error stub
    "gpu_report_oom",                    # OOM reporter
    "gpu_signal_exception",
    "memcpy", "memmove", "memset",
])

GPUCompiler.isintrinsic(::WasmCompilerJob, fn::String) = fn in _whisk_intrinsics
