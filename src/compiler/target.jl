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
