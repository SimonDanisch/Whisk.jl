## Compile + link pipeline
##
## Modeled on Abacus's compiler/compilation.jl, with the final link step
## swapped from a host shared library to a wasm module.

"""
    compile_to_wasm(f, argtypes; out_path, name=nothing, optimize=true)

Compile a Julia function `f(args::argtypes...)` into a `.wasm` module at
`out_path`. Returns a NamedTuple `(; wasm_path, entry_name, llvm_ir)`.

This is the M1 entry point — single function, no allocator, no GC.
"""
function compile_to_wasm(@nospecialize(f), @nospecialize(argtypes::Type{<:Tuple});
                         out_path::String,
                         name::Union{Nothing,String} = nothing,
                         optimize::Bool = true)
    # Phase 1 — Julia source → LLVM IR via GPUCompiler.
    target = WasmCompilerTarget()
    params = WasmCompilerParams()
    config = CompilerConfig(target, params;
                            kernel = false,
                            name   = name,
                            always_inline = false,
                            opt_level = optimize ? 2 : 0)
    src    = methodinstance(typeof(f), argtypes)
    job    = CompilerJob(src, config)

    JuliaContext() do _ctx
        mod, meta = GPUCompiler.compile(:llvm, job)
        entry_fn  = LLVM.name(meta.entry)

        # Mark the entry point as exported so wasm-ld keeps it visible.
        if haskey(LLVM.functions(mod), entry_fn)
            entry_f = LLVM.functions(mod)[entry_fn]
            push!(LLVM.function_attributes(entry_f),
                  LLVM.StringAttribute("wasm-export-name", entry_fn))
        end

        LLVM.verify(mod)

        # Phase 2 — LLVM IR → wasm object file via the LLVM wasm32 backend.
        tm  = GPUCompiler.llvm_machine(target)
        obj = LLVM.emit(tm, mod, LLVM.API.LLVMObjectFile)

        ir_text = string(mod)

        # Phase 3 — wasm object → linked wasm module via lld.
        obj_path = out_path * ".o"
        write(obj_path, obj)
        try
            link_wasm(obj_path, out_path; entry_name = entry_fn)
        finally
            rm(obj_path; force = true)
        end

        return (; wasm_path = out_path, entry_name = entry_fn, llvm_ir = ir_text)
    end
end

"""
    link_wasm(obj_path, wasm_path; entry_name)

Invoke `lld -flavor wasm` to turn a wasm object file into a final wasm
module. Module imports/exports / memory layout are configured here.
"""
function link_wasm(obj_path::String, wasm_path::String; entry_name::String,
                   initial_memory::Integer = 16 * 1024 * 1024,   # 16 MB
                   max_memory::Integer     = 256 * 1024 * 1024)  # 256 MB
    LLD_jll.lld() do lld
        run(`$lld -flavor wasm
             --no-entry
             --export=$entry_name
             --allow-undefined
             --initial-memory=$initial_memory
             --max-memory=$max_memory
             -o $wasm_path
             $obj_path`)
    end
    return wasm_path
end

"""
    methodinstance(ft, tt)

Resolve `Tuple{ft, tt.parameters...}` to a `MethodInstance`. Mirrors
GPUCompiler's internal helper but lives here so we don't depend on its
non-exported API.
"""
function methodinstance(@nospecialize(ft), @nospecialize(tt))
    sig = Tuple{ft, tt.parameters...}
    match = Base._which(sig)
    return Core.Compiler.specialize_method(match)
end
