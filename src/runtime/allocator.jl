## Allocator — M3 (in-wasm bump) → future mark-sweep.
##
## We emit a small, self-contained allocator into every compiled wasm
## module. No JS-side imports, no Runtime.compile registration, no
## global state shared with other GPU compilers.
##
## Layout of linear memory after wasm-ld:
##
##     0  ┌────────────────────────────────┐
##        │ data segments (rodata, etc.)   │
##        ├────────────────────────────────┤
##        │ stack (grows down from sp)     │
##  __heap_base ───────────────────────────┤
##        │ heap (bump-allocated)          │
##        │   ↓                            │
##     ↓  └────────────────────────────────┘
##
## We declare `__heap_base` as an extern symbol; wasm-ld defines it
## and our allocator initialises `_whisk_heap_next` to its address on
## first call. Subsequent allocations bump the heap pointer.

"""
    emit_wasm_runtime!(mod::LLVM.Module)

Inject the in-wasm allocator into `mod`:

  * `@__heap_base`     — extern, supplied by wasm-ld (the byte right
                         after the wasm stack).
  * `@_whisk_heap_next` — i32 global, lazily initialised to
                          `&__heap_base` on first allocation.
  * `whisk_gc_alloc(sz: i32) -> i32`
                          bump allocator, 8-byte aligned. Reads / writes
                          `_whisk_heap_next`. Returns the old value
                          (start of the freshly carved chunk).
  * `jl_alloc_genericmemory_unchecked(_ptls, len_bytes, _type) -> i32`
                          allocates an 8-byte `Memory{T}` struct + a
                          separate `len_bytes` data buffer; writes the
                          buffer pointer at offset 4 of the struct;
                          returns the struct address. The caller (Julia
                          codegen) writes the length at offset 0.

`finish_module!` calls this before any other lowering so the rest of
the pipeline sees a complete, self-contained module.
"""
function emit_wasm_runtime!(mod::LLVM.Module)
    T_i32  = LLVM.Int32Type()
    T_ptr8 = LLVM.PointerType(LLVM.Int8Type())     # opaque ptr in modern LLVM
    T_ptr32 = LLVM.PointerType(T_i32)

    # ── globals
    heap_base = if haskey(LLVM.globals(mod), "__heap_base")
        LLVM.globals(mod)["__heap_base"]
    else
        gv = LLVM.GlobalVariable(mod, LLVM.Int8Type(), "__heap_base")
        LLVM.linkage!(gv, LLVM.API.LLVMExternalLinkage)
        LLVM.extinit!(gv, true)
        gv
    end

    heap_next = if haskey(LLVM.globals(mod), "_whisk_heap_next")
        LLVM.globals(mod)["_whisk_heap_next"]
    else
        gv = LLVM.GlobalVariable(mod, T_i32, "_whisk_heap_next")
        LLVM.initializer!(gv, LLVM.ConstantInt(T_i32, 0))
        LLVM.linkage!(gv, LLVM.API.LLVMInternalLinkage)
        gv
    end

    # ── whisk_gc_alloc
    if !haskey(LLVM.functions(mod), "whisk_gc_alloc") ||
       LLVM.isdeclaration(LLVM.functions(mod)["whisk_gc_alloc"])
        ft = LLVM.FunctionType(T_i32, [T_i32])
        fn = if haskey(LLVM.functions(mod), "whisk_gc_alloc")
            LLVM.functions(mod)["whisk_gc_alloc"]
        else
            LLVM.Function(mod, "whisk_gc_alloc", ft)
        end
        _emit_bump_alloc_body!(fn, heap_base, heap_next)
    end

    # ── jl_alloc_genericmemory_unchecked. Julia's codegen emits this as
    # an extern declaration with `ptr addrspace(10)` return; we match
    # that so the body's terminator type-checks.
    if !haskey(LLVM.functions(mod), "jl_alloc_genericmemory_unchecked") ||
       LLVM.isdeclaration(LLVM.functions(mod)["jl_alloc_genericmemory_unchecked"])
        fn = if haskey(LLVM.functions(mod), "jl_alloc_genericmemory_unchecked")
            LLVM.functions(mod)["jl_alloc_genericmemory_unchecked"]
        else
            ft = LLVM.FunctionType(T_i32, [T_i32, T_i32, T_i32])
            LLVM.Function(mod, "jl_alloc_genericmemory_unchecked", ft)
        end
        _emit_genericmemory_body!(fn, mod)
    end
end

# Bump-allocator body. On first call (heap_next == 0) initialises to
# &__heap_base, then for every call increments by `(sz + 7) & ~7`.
function _emit_bump_alloc_body!(fn::LLVM.Function, heap_base, heap_next)
    LLVM.linkage!(fn, LLVM.API.LLVMInternalLinkage)
    push!(LLVM.function_attributes(fn), LLVM.EnumAttribute("alwaysinline", 0))

    T_i32 = LLVM.Int32Type()
    sz = LLVM.parameters(fn)[1]
    entry = LLVM.BasicBlock(fn, "entry")

    @dispose b=LLVM.IRBuilder() begin
        LLVM.position!(b, entry)

        # cur = load _whisk_heap_next
        cur = LLVM.load!(b, T_i32, heap_next)

        # actual_cur = (cur == 0) ? (i32) &__heap_base : cur
        is_zero = LLVM.icmp!(b, LLVM.API.LLVMIntEQ, cur, LLVM.ConstantInt(T_i32, 0))
        base_addr = LLVM.ptrtoint!(b, heap_base, T_i32)
        actual_cur = LLVM.select!(b, is_zero, base_addr, cur)

        # aligned = (sz + 7) & ~7
        sz7 = LLVM.add!(b, sz, LLVM.ConstantInt(T_i32, 7))
        aligned = LLVM.and!(b, sz7, LLVM.ConstantInt(T_i32, reinterpret(Int32, ~UInt32(7))))

        # next = actual_cur + aligned
        next = LLVM.add!(b, actual_cur, aligned)

        # store next, _whisk_heap_next
        LLVM.store!(b, next, heap_next)

        LLVM.ret!(b, actual_cur)
    end
end

# Memory{T} allocator. Returns a pointer to an 8-byte struct
#   { i32 length; i32 data_ptr; }
# whose `data_ptr` field points to a freshly-carved `len_bytes` buffer.
function _emit_genericmemory_body!(fn::LLVM.Function, mod::LLVM.Module)
    LLVM.linkage!(fn, LLVM.API.LLVMInternalLinkage)
    push!(LLVM.function_attributes(fn), LLVM.EnumAttribute("alwaysinline", 0))

    T_i32 = LLVM.Int32Type()
    T_i8p = LLVM.PointerType(LLVM.Int8Type())
    T_ret = LLVM.return_type(LLVM.function_type(fn))   # may be ptr addrspace(N)
    len_bytes = LLVM.parameters(fn)[2]
    entry = LLVM.BasicBlock(fn, "entry")

    whisk_alloc = LLVM.functions(mod)["whisk_gc_alloc"]
    whisk_alloc_ft = LLVM.function_type(whisk_alloc)

    @dispose b=LLVM.IRBuilder() begin
        LLVM.position!(b, entry)

        ms_addr = LLVM.call!(b, whisk_alloc_ft, whisk_alloc,
                             [LLVM.ConstantInt(T_i32, 8)])
        data_addr = LLVM.call!(b, whisk_alloc_ft, whisk_alloc, [len_bytes])

        # Store data_addr at ms_addr+4 (the struct's `data` field).
        ms_ptr   = LLVM.inttoptr!(b, ms_addr, T_i8p)
        slot_ptr = LLVM.gep!(b, LLVM.Int8Type(), ms_ptr, [LLVM.ConstantInt(T_i32, 4)])
        LLVM.store!(b, data_addr, slot_ptr)

        # Return as the function's declared return type. Julia codegen
        # typically expects `ptr addrspace(10)` here; cast via the
        # int → ptr (addrspace 0) → addrspacecast path for the same
        # reason as above.
        retval = if T_ret isa LLVM.IntegerType
            ms_addr
        else
            T_ptr0 = LLVM.PointerType(LLVM.Int8Type())
            ms_ptr0 = LLVM.inttoptr!(b, ms_addr, T_ptr0)
            LLVM.addrspacecast!(b, ms_ptr0, T_ret)
        end
        LLVM.ret!(b, retval)
    end
end

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

    whisk_alloc    = LLVM.functions(mod)["whisk_gc_alloc"]    # emitted upstream
    whisk_alloc_ft = LLVM.function_type(whisk_alloc)

    T_ptr0 = LLVM.PointerType(LLVM.Int8Type())   # ptr addrspace(0)

    for use in collect(LLVM.uses(alloc_obj))
        call = LLVM.user(use)::LLVM.CallInst
        ops  = LLVM.arguments(call)
        sz   = ops[2]   # already i32 on wasm32

        @dispose builder=LLVM.IRBuilder() begin
            LLVM.position!(builder, call)
            i32_addr = LLVM.call!(builder, whisk_alloc_ft, whisk_alloc, [sz])
            # Julia's GC invariant verifier rejects `inttoptr i32 → ptr
            # addrspace(10)` directly (tracked GC pointers must originate
            # from approved sources). Path through addrspace(0) instead:
            # int → ptr addrspace(0) → ptr addrspace(10) via addrspacecast,
            # which IS allowed.
            as_ptr0 = LLVM.inttoptr!(builder, i32_addr, T_ptr0)
            as_ptr  = LLVM.addrspacecast!(builder, as_ptr0, T_ret)
            LLVM.replace_uses!(call, as_ptr)
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
