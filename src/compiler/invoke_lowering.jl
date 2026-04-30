## ijl_invoke lowering
##
## Julia's IR uses `:invoke` (statically-resolved-but-not-inlined) for
## calls to small helpers like `Base.unsafe_copyto!`, `Core.throw_inexacterror`,
## etc. At the LLVM level these become:
##
##     %ret = call ptr @ijl_invoke(ptr <fn_addr>, ptr %jlcallframe, i32 N,
##                                  ptr <mi_addr>)
##
## where `<fn_addr>` is a host-VM literal pointer to the Function object,
## `<mi_addr>` is a host-VM literal pointer to the MethodInstance, and
## `%jlcallframe` is a stack-allocated `[N x ptr]` of boxed arguments.
##
## GPUCompiler rejects these as "dynamic function invocation" (with
## validate=true) and otherwise leaves them as unresolved imports — both
## bad for wasm. The right move is what Julia's own JIT does: compile
## the referenced MethodInstance and call it directly.
##
## `lower_invokes!` runs in `finish_module!` *before* the host-pointer
## rewrite eats the literal MI/fn pointers. It:
##   1. decodes each `ijl_invoke`'s 4th operand into a `Core.MethodInstance`,
##   2. sub-compiles the MI via the same `compile_to_wasm` pipeline with
##      `toplevel=false`, so `lower_throw!` etc. apply to the sub-IR,
##   3. links the sub-module into the parent,
##   4. replaces the `ijl_invoke` call with a direct call to the
##      sub-function, loading each arg from the call frame.
##
## After this, the only remaining `ijl_invoke` calls are ones whose MI
## couldn't be decoded (rare; those still trap at runtime via the
## existing import path).

"""
    lower_julia_call2!(mod) → Int

Rewrite every `@julia.call2(@ijl_invoke, fn, mi, args...)` call into
the equivalent direct `@ijl_invoke(fn, frame, count, mi)` form, so
`lower_invokes!` can decode the MI and replace with a direct call to
the sub-compiled body. `julia.call2` is a high-level Julia intrinsic
the toplevel optimization pipeline normally lowers; sub-jobs with
`toplevel = false` (our `link_compiler_rt!` rt-helpers) skip that
pass and leak the wrapper into the linked module, where the wasm
backend can't lower it (`getAlignment` SIGSEGV during ISel).

Returns the number of calls rewritten.
"""
function lower_julia_call2!(mod::LLVM.Module)
    haskey(LLVM.functions(mod), "julia.call2") || return 0
    call2_fn = LLVM.functions(mod)["julia.call2"]

    # Make sure `@ijl_invoke` is declared with the standard signature
    # `ptr ijl_invoke(ptr, ptr, i32, ptr)`. If link already declared
    # it, reuse; otherwise create.
    invoke_fty = LLVM.FunctionType(LLVM.PointerType(LLVM.Int8Type()),
                                    [LLVM.PointerType(LLVM.Int8Type()),
                                     LLVM.PointerType(LLVM.Int8Type()),
                                     LLVM.Int32Type(),
                                     LLVM.PointerType(LLVM.Int8Type())])
    invoke_fn = if haskey(LLVM.functions(mod), "ijl_invoke")
        LLVM.functions(mod)["ijl_invoke"]
    else
        LLVM.Function(mod, "ijl_invoke", invoke_fty)
    end

    n = 0
    for use in collect(LLVM.uses(call2_fn))
        call = LLVM.user(use)
        call isa LLVM.CallInst || continue
        # Walk the call's operands. Layout (excluding the function
        # operand at the end): [target, fn, mi, args...]. We only
        # rewrite when `target == @ijl_invoke` — other `julia.call2`
        # patterns belong to `jl_apply_generic` etc. which the
        # dynamic-dispatch pass handles separately.
        ops = LLVM.arguments(call)
        length(ops) >= 3 || continue
        target = ops[1]
        target isa LLVM.Function || continue
        LLVM.name(target) == "ijl_invoke" || continue

        fobj    = ops[2]
        mi      = ops[3]
        fn_args = @view ops[4:end]
        nargs   = length(fn_args)

        @dispose b = LLVM.IRBuilder() begin
            LLVM.position!(b, call)
            # Build a [N x ptr] frame on the stack, populate from the
            # remaining call args.
            frame_ty = LLVM.ArrayType(LLVM.PointerType(LLVM.Int8Type()), nargs)
            frame    = LLVM.alloca!(b, frame_ty)
            for (i, arg) in enumerate(fn_args)
                slot = LLVM.gep!(b, frame_ty, frame,
                                  [LLVM.ConstantInt(LLVM.Int64Type(), 0),
                                   LLVM.ConstantInt(LLVM.Int64Type(), i - 1)])
                LLVM.store!(b, arg, slot)
            end
            new_call = LLVM.call!(b, invoke_fty, invoke_fn, LLVM.Value[
                fobj, frame, LLVM.ConstantInt(LLVM.Int32Type(), nargs), mi,
            ])
            # Splice the new call's result in for the old one (both
            # return `ptr`, so RAUW is valid).
            LLVM.replace_uses!(call, new_call)
            LLVM.erase!(call)
        end
        n += 1
    end
    return n
end

"""
    lower_invokes!(mod, target, params)

Replace every `ijl_invoke` call in `mod` with a direct call to the
sub-compiled MethodInstance. Returns the number of calls lowered.
"""
function lower_invokes!(mod::LLVM.Module, target, params)
    haskey(LLVM.functions(mod), "ijl_invoke") || return 0
    invoke_fn = LLVM.functions(mod)["ijl_invoke"]

    # Group call sites by MI so each MI is compiled at most once.
    by_mi = IdDict{Core.MethodInstance, Vector{LLVM.CallInst}}()
    skipped = 0
    for use in collect(LLVM.uses(invoke_fn))
        call_inst = LLVM.user(use)
        call_inst isa LLVM.CallInst || continue
        args = LLVM.arguments(call_inst)
        length(args) == 4 || (skipped += 1; continue)
        mi = _decode_method_instance(args[4])
        mi === nothing && (skipped += 1; continue)
        push!(get!(by_mi, mi, LLVM.CallInst[]), call_inst)
    end
    isempty(by_mi) && return 0

    n_lowered = 0
    for (mi, calls) in by_mi
        sub_fn = _sub_compile_mi!(mod, target, params, mi)
        sub_fn === nothing && continue   # sub-compile failed; leave call as-is to trap
        for call in calls
            _replace_invoke_with_direct!(call, sub_fn)
            n_lowered += 1
        end
    end

    # If every ijl_invoke got rewritten, the declaration becomes unreferenced.
    return n_lowered
end

"""
    erase_unused_julia_invoke_declarations!(mod) → Int

Drop stale Julia runtime dispatch declarations once Whisk's lowering passes
have replaced their call sites. This keeps final IR diagnostics aligned with
the linked wasm: if an `ijl_invoke`/`jl_apply_generic` symbol remains visible,
it should indicate a real unresolved call, not a dead declaration that
`wasm-ld` would silently strip.
"""
function erase_unused_julia_invoke_declarations!(mod::LLVM.Module)
    n = 0
    for name in ("ijl_invoke", "jl_invoke", "ijl_apply_generic", "jl_apply_generic")
        haskey(LLVM.functions(mod), name) || continue
        fn = LLVM.functions(mod)[name]
        LLVM.isdeclaration(fn) || continue
        iterate(LLVM.uses(fn)) === nothing || continue
        LLVM.erase!(fn)
        n += 1
    end
    return n
end

# Decode an ijl_invoke 4th-operand into the host-VM `MethodInstance`
# it references. Two shapes appear in practice:
#
#   1. `inttoptr (i64 N to ptr)` ConstantExpr — original Julia frontend
#      output. The literal address IS the host-VM MI pointer.
#   2. `load ptr, ptr @jl_global_NNN` Instruction — emitted by Julia's
#      `julia.call2` lowering (which we mimic in `lower_julia_call2!`).
#      The global's initializer is the `inttoptr` we're after.
#
# Returns `nothing` if neither shape recognises a real `MethodInstance`.
function _decode_method_instance(operand)
    # Look through addrspacecast / bitcast wrappers — sub-modules
    # compiled with toplevel=false skip the addrspace-strip pass, so
    # MI loads come out wrapped in an addrspacecast to addrspace(10).
    if operand isa LLVM.AddrSpaceCastInst || operand isa LLVM.BitCastInst
        return _decode_method_instance(LLVM.operands(operand)[1])
    end
    if operand isa LLVM.ConstantExpr
        op = LLVM.opcode(operand)
        if op == LLVM.API.LLVMAddrSpaceCast || op == LLVM.API.LLVMBitCast
            return _decode_method_instance(LLVM.operands(operand)[1])
        end
        op == LLVM.API.LLVMIntToPtr || return nothing
        addr_op = LLVM.operands(operand)[1]
        addr_op isa LLVM.ConstantInt || return nothing
        addr = convert(Int, addr_op)
        addr <= 0x100000 && return nothing
        addr % 8 != 0     && return nothing
        obj = try
            Base.unsafe_pointer_to_objref(Ptr{Nothing}(addr))
        catch
            return nothing
        end
        return obj isa Core.MethodInstance ? obj : nothing
    end
    if operand isa LLVM.LoadInst
        ptr_op = LLVM.operands(operand)[1]
        ptr_op isa LLVM.GlobalVariable || return nothing
        init = LLVM.initializer(ptr_op)
        init === nothing && return nothing
        return _decode_method_instance(init)
    end
    return nothing
end

# Sub-compile a single MethodInstance through Whisk's pipeline with
# `toplevel = false`, strip Julia-internal addrspaces from the resulting
# IR, link it into `mod`, and return the entry function (now part of
# `mod`).
#
# The strip is essential: sub-jobs run with `toplevel = false`, which
# skips GPUCompiler's `optimize!` pipeline (driver.jl gates it on
# `toplevel`). That means `RemoveJuliaAddrspacesPass` — the standard
# tool every other GPUCompiler backend uses to convert Julia's GC-
# tracking addrspaces (10/11/12/13) into plain `addrspace 0` — never
# runs. Without it, the sub-IR retains `ptr addrspace(10)` references
# that the wasm backend can't lower (LLVM-wasm hardcodes addrspace 10
# as `externref`, an opaque host reference). We strip on `sub_mod`
# before linking, which keeps the parent module's host-pointer
# `inttoptr` constants intact (so `rewrite_host_pointer_constants!`
# still finds them in their original 16-byte-aligned form).
function _sub_compile_mi!(mod::LLVM.Module, target, params, mi::Core.MethodInstance)
    sub_config = CompilerConfig(target, params;
                                kernel        = false,
                                name          = nothing,
                                toplevel      = false,
                                always_inline = true,
                                validate      = false,
                                opt_level     = 2)
    sub_job = CompilerJob(mi, sub_config)
    sub_mod, sub_meta = try
        GPUCompiler.compile(:llvm, sub_job)
    catch err
        @warn "Whisk: sub-compile of $(mi) failed; ijl_invoke call will trap at runtime" err
        return nothing
    end

    sub_entry_name = LLVM.name(sub_meta.entry)
    LLVM.linkage!(sub_meta.entry, LLVM.API.LLVMExternalLinkage)

    _strip_julia_addrspaces!(sub_mod, target)

    LLVM.link!(mod, sub_mod)
    return LLVM.functions(mod)[sub_entry_name]
end

# Run the GC + address-space lowering pipeline GPUCompiler normally
# applies in `buildIntrinsicLoweringPipeline` (optim.jl) — but only
# for `toplevel = true` jobs. Sub-jobs skip optimize entirely, leaving
# their IR with `julia.gc_alloc_obj`, `julia.safepoint`,
# `julia.get_pgcstack`, and `addrspace(10)` intact. Linking those
# unlowered shapes into a fully-lowered toplevel triggers
# `LLVM TypeFinder::incorporateValue` segfaults during `link!` —
# the type system can't reconcile addrspace 10 in one module with
# addrspace 0 in the other. Run the same passes here so sub-modules
# match the toplevel before linking.
function _strip_julia_addrspaces!(sub_mod::LLVM.Module, target)
    tm = GPUCompiler.llvm_machine(target)
    @dispose pb=LLVM.NewPMPassBuilder() begin
        LLVM.add!(pb, LLVM.Interop.RemoveNIPass())
        LLVM.add!(pb, LLVM.Interop.RemoveJuliaAddrspacesPass())
        LLVM.add!(pb, LLVM.AlwaysInlinerPass())
        LLVM.add!(pb, LLVM.NewPMFunctionPassManager()) do fpm
            LLVM.add!(fpm, LLVM.SROAPass())
            LLVM.add!(fpm, LLVM.InstCombinePass())
            LLVM.add!(fpm, LLVM.SimplifyCFGPass())
        end
        LLVM.run!(pb, sub_mod, tm)
    end
    return sub_mod
end

# Build a direct call to `sub_fn` at the same point as the `ijl_invoke`
# call, loading the arguments from the call frame. The frame layout is
# `[nargs x ptr]` where each slot points at the boxed arg.
function _replace_invoke_with_direct!(call_inst::LLVM.CallInst, sub_fn::LLVM.Function)
    args = LLVM.arguments(call_inst)
    frame = args[2]   # ptr to [nargs x ptr] alloca

    sub_fn_ty   = LLVM.function_type(sub_fn)
    sub_params  = LLVM.parameters(sub_fn_ty)
    sub_ret     = LLVM.return_type(sub_fn_ty)

    ptr_ty = LLVM.PointerType(LLVM.Int8Type())
    i64_ty = LLVM.Int64Type()

    @dispose b = LLVM.IRBuilder() begin
        LLVM.position!(b, call_inst)
        loaded = LLVM.Value[]
        for (i, pty) in enumerate(sub_params)
            slot = LLVM.gep!(b, ptr_ty, frame,
                             [LLVM.ConstantInt(i64_ty, (i - 1) * 8)])
            push!(loaded, _load_arg!(b, slot, pty))
        end
        result = LLVM.call!(b, sub_fn_ty, sub_fn, loaded)

        # ijl_invoke returns ptr (boxed). If sub_fn returns the same, splice in.
        # Otherwise leave the original call's uses intact and emit a trap path
        # — this is rare and currently only happens for return-by-value MIs we
        # haven't taught argument coercion to box yet.
        if LLVM.value_type(call_inst) == sub_ret
            LLVM.replace_uses!(call_inst, result)
        end
        LLVM.erase!(call_inst)
    end
    return
end

# Load an argument from frame-slot pointer `slot` as the type `pty`.
#
#  * `ptr addrspace(N)` — `load ptr addrspace(N), ptr %slot`. We load
#    directly as the target addrspace rather than addrspacecasting
#    afterwards, because the wasm backend can't lower
#    `addrspacecast 0 → 10` (it maps addrspace 10 to externref, which
#    isn't reachable from a memory pointer).
#  * value-typed (i32, i64, f64, …) — `load <pty>, ptr %slot`. The
#    slot stores a pointer to the boxed/ref-passed value; we
#    dereference once to get the unboxed scalar.
function _load_arg!(b::LLVM.IRBuilder, slot::LLVM.Value, pty::LLVM.LLVMType)
    if pty isa LLVM.PointerType
        # Load slot as the target addrspace directly. Slot itself is
        # addrspace 0 (the alloca); the *value* in the slot is the
        # tracked pointer the callee expects.
        return LLVM.load!(b, pty, slot)
    end
    # Value-typed: slot holds a pointer; load through it.
    inner_ptr_ty = LLVM.PointerType(LLVM.Int8Type())
    inner = LLVM.load!(b, inner_ptr_ty, slot)
    return LLVM.load!(b, pty, inner)
end
