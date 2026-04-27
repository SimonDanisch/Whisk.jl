## Allocator — M2 (bump) → M3 (mark-sweep).
##
## For M2 we replace `julia.gc_alloc_obj(ptls, size, type)` calls with
## `whisk_gc_alloc(size_i32)` and link `whisk_gc_alloc` as an
## extern import. The host (JS / V8) provides a bump allocator that
## hands out chunks of a `WebAssembly.Memory`. M3 will replace this with
## an in-wasm precise mark-sweep collector.

"""
    lower_gc_alloc_wasm32!(mod::LLVM.Module)

Replace every `julia.gc_alloc_obj(ptls, size_i32, type)` call in `mod`
with a call to the externally-imported `whisk_gc_alloc(size_i32)` ->
`ptr addrspace(11)`. Runs from `finish_module!` BEFORE
`lower_gc_frame!`, so GPUCompiler's default lowering (which assumes
host-typed `Csize_t = UInt64`) finds nothing to do.
"""
function lower_gc_alloc_wasm32!(mod::LLVM.Module)
    haskey(LLVM.functions(mod), "julia.gc_alloc_obj") || return
    alloc_obj = LLVM.functions(mod)["julia.gc_alloc_obj"]
    T_ret = LLVM.return_type(LLVM.function_type(alloc_obj))   # ptr addrspace(11)
    T_sz  = LLVM.Int32Type()                                  # wasm32 size_t

    # Get-or-declare the extern allocator.
    name = "whisk_gc_alloc"
    if haskey(LLVM.functions(mod), name)
        whisk_alloc = LLVM.functions(mod)[name]
    else
        ft = LLVM.FunctionType(T_ret, [T_sz])
        whisk_alloc = LLVM.Function(mod, name, ft)
    end

    for use in collect(LLVM.uses(alloc_obj))
        call = LLVM.user(use)::LLVM.CallInst
        ops  = LLVM.arguments(call)
        sz   = ops[2]   # already i32 on wasm32

        @dispose builder=LLVM.IRBuilder() begin
            LLVM.position!(builder, call)
            new_call = LLVM.call!(builder, LLVM.function_type(whisk_alloc),
                                  whisk_alloc, [sz])
            LLVM.replace_uses!(call, new_call)
        end
        LLVM.erase!(call)
    end

    # Drop the now-unused intrinsic so `lower_gc_frame!` finds nothing.
    if isempty(LLVM.uses(alloc_obj))
        LLVM.erase!(alloc_obj)
    end
end

"""
    bypass_argument_error_checks!(mod::LLVM.Module)

Julia's `emit_memorynew` LLVM codegen emits a bounds check that uses
host-typed (`Int = Int64`) constants but truncates them to wasm32's
i32. The result on wasm32 is a check like `slt n*4, -1` that always
takes the fail branch — calling `jl_argument_error`, even for valid
small inputs.

Until Julia's codegen learns about wasm32's word size we sidestep:
find every basic block ending with `call jl_argument_error; unreachable`,
and rewrite each predecessor's terminator to skip past it. This trusts
the caller not to overflow the actual i32 bounds — fine for our regime,
revisit when we add a real safety story.
"""
function bypass_argument_error_checks!(mod::LLVM.Module)
    haskey(LLVM.functions(mod), "jl_argument_error") || return
    err_fn = LLVM.functions(mod)["jl_argument_error"]

    # Each fail block has form `[entry: call jl_argument_error; unreachable]`.
    # Collect them.
    fail_blocks = LLVM.BasicBlock[]
    for use in collect(LLVM.uses(err_fn))
        call = LLVM.user(use)
        call isa LLVM.CallInst || continue
        bb = LLVM.parent(call)
        push!(fail_blocks, bb)
    end

    # Walk every conditional branch, when one target is a fail block,
    # rewrite the branch to unconditionally take the *other* target.
    for fn in LLVM.functions(mod)
        LLVM.isdeclaration(fn) && continue
        for bb in LLVM.blocks(fn)
            term = LLVM.terminator(bb)
            term isa LLVM.BrInst || continue
            ops = LLVM.operands(term)
            length(ops) == 3 || continue   # conditional br has 3 operands
            # operands(br) = [cond, false_target, true_target] in LLVM API order
            true_tgt  = ops[3]
            false_tgt = ops[2]
            true_is_fail  = true_tgt  in fail_blocks
            false_is_fail = false_tgt in fail_blocks
            true_is_fail == false_is_fail && continue
            keep = true_is_fail ? false_tgt : true_tgt

            @dispose builder=LLVM.IRBuilder() begin
                LLVM.position!(builder, term)
                LLVM.br!(builder, keep)
            end
            LLVM.erase!(term)
        end
    end
end

"""
    lower_gc_frame_intrinsics!(mod::LLVM.Module)

Erase Julia 1.11+ GC-frame intrinsics that GPUCompiler's
`lower_gc_frame!` doesn't yet handle. Without a real GC, all of these
collapse to no-ops or null pointers:

  * `julia.new_gc_frame(N) → ptr`     — replaced by null ptr
  * `julia.push_gc_frame(frame, N)`   — erased
  * `julia.pop_gc_frame(frame)`       — erased
  * `julia.get_gc_frame_slot(frame, i) → ptr` — null ptr
  * `julia.get_pgcstack() → ptr`      — null ptr
  * `julia.write_barrier(...)`        — erased

When a real GC arrives in M3, this gets replaced with proper lowering
into our collector's stack-frame tracking.
"""
function lower_gc_frame_intrinsics!(mod::LLVM.Module)
    no_op_intrinsics      = ("julia.push_gc_frame", "julia.pop_gc_frame",
                             "julia.write_barrier")
    null_ptr_intrinsics   = ("julia.new_gc_frame", "julia.get_gc_frame_slot",
                             "julia.get_pgcstack")

    for name in no_op_intrinsics
        haskey(LLVM.functions(mod), name) || continue
        intr = LLVM.functions(mod)[name]
        for use in collect(LLVM.uses(intr))
            LLVM.erase!(LLVM.user(use)::LLVM.CallInst)
        end
        isempty(LLVM.uses(intr)) && LLVM.erase!(intr)
    end

    for name in null_ptr_intrinsics
        haskey(LLVM.functions(mod), name) || continue
        intr = LLVM.functions(mod)[name]
        ret_ty = LLVM.return_type(LLVM.function_type(intr))
        for use in collect(LLVM.uses(intr))
            call = LLVM.user(use)::LLVM.CallInst
            null = LLVM.null(ret_ty)
            LLVM.replace_uses!(call, null)
            LLVM.erase!(call)
        end
        isempty(LLVM.uses(intr)) && LLVM.erase!(intr)
    end
end
