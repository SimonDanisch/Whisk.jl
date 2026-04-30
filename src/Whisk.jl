module Whisk

using GPUCompiler
using LLVM
using LLVM.Interop
using Libdl
using Pkg
using Zstd_jll

# WebAssembly target backends are present in Julia's bundled LLVM but not
# initialised by default (only X86 is). Force-init at module load so
# `LLVM.Target(triple="wasm32-unknown-unknown")` works.
function __init__()
    LLVM.InitializeWebAssemblyTarget()
    LLVM.InitializeWebAssemblyTargetInfo()
    LLVM.InitializeWebAssemblyTargetMC()
    LLVM.InitializeWebAssemblyAsmPrinter()
    LLVM.InitializeWebAssemblyAsmParser()
end

# --- compiler ----------------------------------------------------------
include("compiler/host_globals.jl")
include("compiler/diagnostics.jl")
include("compiler/wasm_validator.jl")
include("compiler/target.jl")
include("compiler/invoke_lowering.jl")
include("compiler/call_indirect.jl")
include("compiler/compilation.jl")
include("compiler/validation.jl")
include("compiler/reflection.jl")

# --- runtime (compiled into every output module) -----------------------
include("runtime/intrinsics.jl")
include("runtime/layout.jl")
include("runtime/gc.jl")
include("runtime/exceptions.jl")
include("runtime/runtime.jl")
include("runtime/handles.jl")
include("runtime/layout_addr_table.jl")

# --- Julia stdlib overrides --------------------------------------------
include("quirks.jl")

# --- JS interop --------------------------------------------------------
include("jsinterop/marshal.jl")
include("jsinterop/jsexport.jl")
include("jsinterop/image_builder.jl")

# --- Generic wasm-boundary marshal ------------------------------------
include("marshal.jl")

export compile, @code_wasm, @jsexport, UnsupportedCcallError

end # module Whisk
