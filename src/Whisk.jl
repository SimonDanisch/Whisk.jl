module Whisk

using GPUCompiler
using LLVM
using LLVM.Interop
using LLD_jll
using Libdl

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
include("compiler/target.jl")
include("compiler/compilation.jl")
include("compiler/validation.jl")
include("compiler/reflection.jl")

# --- runtime (compiled into every output module) -----------------------
include("runtime/intrinsics.jl")
include("runtime/allocator.jl")
include("runtime/exceptions.jl")
include("runtime/runtime.jl")

# --- Julia stdlib overrides --------------------------------------------
include("quirks.jl")

# --- JS interop --------------------------------------------------------
include("jsinterop/marshal.jl")
include("jsinterop/jsexport.jl")

export compile, @code_wasm, @jsexport, UnsupportedCcallError

end # module Whisk
