## wasm call_indirect support
##
## This file starts with a deliberately small probe used by M13. It proves the
## LLVM IR shape Whisk needs for MethodInstance vtables: a function address is
## stored as a wasm64 table index, loaded, converted to a callable pointer, and
## called through a non-direct callee. The general vtable lowering can reuse
## this shape.

"""
    emit_call_indirect_probe!(mod) → LLVM.Function

Emit an exported `whisk_call_indirect_probe(i32) -> i32` function that calls
an internal target through a loaded function pointer. The target returns
`x + 1`, so the exported probe is easy to validate from JS.
"""
function emit_call_indirect_probe!(mod::LLVM.Module)
    i32 = LLVM.Int32Type()
    i64 = LLVM.Int64Type()
    ptr = LLVM.PointerType(LLVM.Int8Type())
    fty = LLVM.FunctionType(i32, [i32])

    target = if haskey(LLVM.functions(mod), "whisk_call_indirect_probe_target")
        LLVM.functions(mod)["whisk_call_indirect_probe_target"]
    else
        fn = LLVM.Function(mod, "whisk_call_indirect_probe_target", fty)
        LLVM.linkage!(fn, LLVM.API.LLVMInternalLinkage)
        bb = LLVM.BasicBlock(fn, "entry")
        @dispose b = LLVM.IRBuilder() begin
            LLVM.position!(b, bb)
            x = LLVM.parameters(fn)[1]
            y = LLVM.add!(b, x, LLVM.ConstantInt(i32, 1))
            LLVM.ret!(b, y)
        end
        fn
    end

    slot = if haskey(LLVM.globals(mod), "whisk_call_indirect_probe_slot")
        LLVM.globals(mod)["whisk_call_indirect_probe_slot"]
    else
        gv = LLVM.GlobalVariable(mod, i64, "whisk_call_indirect_probe_slot")
        LLVM.initializer!(gv, LLVM.const_ptrtoint(target, i64))
        LLVM.linkage!(gv, LLVM.API.LLVMInternalLinkage)
        LLVM.constant!(gv, true)
        gv
    end

    probe = if haskey(LLVM.functions(mod), "whisk_call_indirect_probe")
        LLVM.functions(mod)["whisk_call_indirect_probe"]
    else
        fn = LLVM.Function(mod, "whisk_call_indirect_probe", fty)
        LLVM.linkage!(fn, LLVM.API.LLVMExternalLinkage)
        push!(LLVM.function_attributes(fn),
              LLVM.StringAttribute("wasm-export-name", "whisk_call_indirect_probe"))
        bb = LLVM.BasicBlock(fn, "entry")
        @dispose b = LLVM.IRBuilder() begin
            LLVM.position!(b, bb)
            idx = LLVM.load!(b, i64, slot)
            fp = LLVM.inttoptr!(b, idx, ptr)
            call = LLVM.call!(b, fty, fp, LLVM.Value[LLVM.parameters(fn)[1]])
            LLVM.ret!(b, call)
        end
        fn
    end

    LLVM.API.LLVMAppendToUsed(mod, [target, probe], 2)
    return probe
end
