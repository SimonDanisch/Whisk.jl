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
            _replace_invoke_with_direct!(call, sub_fn, mi)
            n_lowered += 1
        end
    end

    # If every ijl_invoke got rewritten, the declaration becomes unreferenced.
    # Leave it for now — wasm-ld will dead-strip it.
    return n_lowered
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
# Per-module memo: MI → linked function NAME (names survive module
# passes; handles don't). Without this the lowering fixpoint
# re-sub-compiles the same MIs every iteration and DIVERGES (each
# round links fresh copies carrying fresh unresolved wrappers).
const SUB_COMPILE_MEMO = Dict{UInt64, Dict{Core.MethodInstance, String}}()

function _sub_compile_mi!(mod::LLVM.Module, target, params, mi::Core.MethodInstance)
    memo = get!(() -> Dict{Core.MethodInstance, String}(),
                SUB_COMPILE_MEMO, objectid(mod))
    if haskey(memo, mi)
        nm = memo[mi]
        haskey(LLVM.functions(mod), nm) && return LLVM.functions(mod)[nm]
    end
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
    memo[mi] = sub_entry_name
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
# `[nargs x ptr]` where each slot points at the boxed arg (the callee
# object travels separately as operand 1).
function _replace_invoke_with_direct!(call_inst::LLVM.CallInst, sub_fn::LLVM.Function,
                                      mi::Union{Core.MethodInstance, Nothing} = nothing)
    args = LLVM.arguments(call_inst)
    fval  = args[1]   # callee object
    frame = args[2]   # ptr to [nargs x ptr] alloca

    sub_fn_ty   = LLVM.function_type(sub_fn)
    sub_params  = LLVM.parameters(sub_fn_ty)
    sub_ret     = LLVM.return_type(sub_fn_ty)

    ptr_ty = LLVM.PointerType(LLVM.Int8Type())
    i64_ty = LLVM.Int64Type()

    @dispose b = LLVM.IRBuilder() begin
        LLVM.position!(b, call_inst)
        # Signature-driven slot mapping first: mid-signature ghost
        # arguments (`Type{X}`, singletons) occupy frame slots without
        # LLVM parameters, so a positional walk mis-slots everything
        # after them (`_(::Bool, ::Type{Observable{T}}, val)` passed
        # the Type as `val`). Positional fallback for signatures the
        # mapper can't align (Vararg specialisations).
        loaded = mi === nothing ? nothing :
                 dispatch_call_args!(b, sub_fn, mi, fval, frame, i64_ty)
        if loaded === nothing
            loaded = LLVM.Value[]
            for (i, pty) in enumerate(sub_params)
                # Element-typed GEP: stride is sizeof(ptr) per index
                # unit, so slot i lives at index i-1 — NOT byte offset
                # (i-1)*8, which would scale to (i-1)*64 bytes and read
                # garbage for every argument past the first (silent
                # wrong-args miscompile, found via Dict{Symbol,Any}
                # setindex!).
                slot = LLVM.gep!(b, ptr_ty, frame,
                                 [LLVM.ConstantInt(i64_ty, i - 1)])
                push!(loaded, _load_arg!(b, slot, pty))
            end
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

# Position builder `b` at `call`, inheriting its debug location. When
# the site carries none, synthesize line 0 in the CALLER's subprogram —
# LLVM requires every inlinable call inside a function with debug info
# to carry a `!dbg`.
function position_with_dbg!(b::LLVM.IRBuilder, call::LLVM.Instruction)
    LLVM.position!(b, call)
    LLVM.debuglocation!(b, call)
    if LLVM.debuglocation(b) === nothing
        spc = LLVM.get_subprogram(LLVM.parent(LLVM.parent(call)))
        spc !== nothing && LLVM.debuglocation!(b, LLVM.DILocation(0, 0, spc))
    end
    return b
end

# Does an argument of type `Ti` contribute an LLVM parameter? Mirrors
# codegen's ghost rules: singleton instances (`typeof(f)`, `Nothing`)
# AND uniquely-represented `Type{X}` arguments (is_uniquerep_Type in
# codegen.cpp — `Base.issingletontype` is FALSE for those, they carry
# no `.instance`) are value-less and get no parameter.
function ghost_argtype(@nospecialize(Ti))
    Ti isa DataType || return false
    Base.issingletontype(Ti) && return true
    return Core.Compiler.isconstType(Ti)
end

# Map jlcallframe slots onto a sub-compiled callee's LLVM parameters,
# driven by the METHOD SIGNATURE: ghost-typed arguments (the callee's
# own type, `Type{X}` params, `nothing`) have no LLVM parameter and no
# influence on slot order, but they DO occupy frame slots — a
# positional `has_callee` heuristic mis-slots every argument after a
# mid-signature ghost (`_(::Bool, ::Type{Observable{T}}, val)` loaded
# the Type as `val`). Returns the argument Values, or `nothing` when
# the signature can't be aligned with the LLVM parameter list (exotic
# ABI — caller falls back / skips).
function dispatch_call_args!(b, sub::LLVM.Function, mi::Core.MethodInstance,
                             fval, frame, i64)
    sig = mi.specTypes
    sig isa DataType || return nothing
    tys = sig.parameters
    sub_ps = LLVM.parameters(LLVM.function_type(sub))
    loaded = LLVM.Value[]
    pj = 1
    for (i, Ti) in enumerate(tys)
        ghost_argtype(Ti) && continue
        pj > length(sub_ps) && return nothing
        pty = sub_ps[pj]
        if i == 1
            # Non-ghost callee: boxed → pointer; unboxed closure env
            # passed by value → load the payload.
            push!(loaded, pty isa LLVM.PointerType ? fval :
                  LLVM.load!(b, pty, fval))
        else
            slot = LLVM.gep!(b, LLVM.PointerType(LLVM.Int8Type()), frame,
                             [LLVM.ConstantInt(i64, i - 2)])
            push!(loaded, _load_arg!(b, slot, pty))
        end
        pj += 1
    end
    pj == length(sub_ps) + 1 || return nothing
    return loaded
end

# Boxers for primitive returns: dispatch-table candidates often
# return unboxed scalars while the `jl_apply_generic` contract is a
# boxed pointer. The boxing semantics live in the Julia-implemented
# runtime helpers (`link_compiler_rt!` links them before the lowering
# fixpoint); the emitter only places a call.
const RETURN_BOXERS = Dict{DataType, String}(
    Float64 => "ijl_box_float64", Float32 => "ijl_box_float32",
    Int64   => "ijl_box_int64",   UInt64  => "ijl_box_uint64",
    Int32   => "ijl_box_int32",   UInt32  => "ijl_box_uint32",
    Int16   => "ijl_box_int16",   UInt16  => "ijl_box_uint16",
    Int8    => "ijl_box_int8",    UInt8   => "ijl_box_uint8",
    Bool    => "ijl_box_bool",    Char    => "ijl_box_char",
)

# Inferred return type of `mi` (`nothing` when inference is
# unavailable — caller falls back to an undef result).
function return_type_of(mi::Core.MethodInstance)
    results = try
        Base.code_typed_by_type(mi.specTypes; optimize = false)
    catch
        return nothing
    end
    isempty(results) && return nothing
    return results[1][2]
end

# Box an unboxed scalar `r` returned by dispatch-table candidate `mi`
# into a fresh heap cell via the runtime's `ijl_box_*` helper. Returns
# the boxed ptr Value, or `nothing` when the return type has no boxer
# (struct returns — future work alongside invoke-path return boxing).
function box_primitive_result!(b, mod::LLVM.Module, mi::Core.MethodInstance,
                               r::LLVM.Value)
    jlrt = return_type_of(mi)
    jlrt isa DataType || return nothing
    boxer = get(RETURN_BOXERS, jlrt, nothing)
    boxer === nothing && return nothing
    haskey(LLVM.functions(mod), boxer) || return nothing
    bfn = LLVM.functions(mod)[boxer]
    bft = LLVM.function_type(bfn)
    LLVM.return_type(bft) isa LLVM.PointerType || return nothing
    pty = only(LLVM.parameters(bft))
    v = r
    if LLVM.value_type(v) != pty
        # `i1`→`i8` (Bool) and similar width adjustments; anything
        # non-integral that disagrees is an ABI we don't understand.
        (LLVM.value_type(v) isa LLVM.IntegerType && pty isa LLVM.IntegerType) ||
            return nothing
        vw = LLVM.width(LLVM.value_type(v))
        pw = LLVM.width(pty)
        v = vw < pw ? LLVM.zext!(b, v, pty) :
            vw > pw ? LLVM.trunc!(b, v, pty) : v
    end
    return LLVM.call!(b, bft, bfn, [v])
end

# Recover compile-time-known frame slot values at a jlcall site: scan
# the stores into `frame` that precede `call` in its block and decode
# each stored operand. An entry is `missing` when no store was seen
# and `nothing` when the stored value does not decode to a host
# constant — either way the slot is runtime-only to the caller.
function decode_frame_slots(call::LLVM.CallInst, frame, nargs::Int)
    slot_vals = Vector{Any}(undef, nargs)
    fill!(slot_vals, missing)
    for inst in collect(LLVM.instructions(LLVM.parent(call)))
        inst === call && break
        inst isa LLVM.StoreInst || continue
        ptr = LLVM.operands(inst)[2]
        # Frame slots are addressed either as `gep i8/ptr, %frame, N`
        # (jlcallframe) or `gep [n x ptr], %frame, 0, N` (sub-compiled
        # alloca) — the slot index is the LAST gep index, and any
        # leading indices must be zero.
        idx = if ptr === frame
            0
        elseif ptr isa LLVM.Instruction &&
               LLVM.opcode(ptr) == LLVM.API.LLVMGetElementPtr &&
               LLVM.operands(ptr)[1] === frame &&
               length(LLVM.operands(ptr)) >= 2 &&
               all(op -> op isa LLVM.ConstantInt, LLVM.operands(ptr)[2:end]) &&
               all(op -> convert(Int, op) == 0, LLVM.operands(ptr)[2:end-1])
            convert(Int, LLVM.operands(ptr)[end])
        else
            -1
        end
        0 <= idx < nargs || continue
        slot_vals[idx + 1] = decode_host_object(LLVM.operands(inst)[1])
    end
    return slot_vals
end

"""
    lower_apply_generic!(mod, world, target, params) → Int

Closed-world dynamic dispatch (ARCHITECTURE.md phase 5). For every
`jl_apply_generic`/`ijl_apply_generic` call site: build the candidate
callee set from the reachable `world`, sub-compile each candidate MI,
and replace the call with an objectid-switch —

    tag  = load(f - 8) & ~15          ; unified descriptor
    oid  = load(tag + 16)             ; fixed objectid slot
    switch oid [objectid(Tᵢ) → call subᵢ(f, args…)]   default → trap

Args are loaded from the jlcallframe (same `_load_arg!` coercion as
invoke lowering); the callee object itself is argument 1. Sites with
no candidates keep the loud-trap lowering from
`lower_dynamic_dispatch!` (which must run AFTER this pass).
"""
function lower_apply_generic!(mod::LLVM.Module, world, target, params)
    n = 0
    i64 = LLVM.Int64Type()
    # Candidate enumeration sweeps the universe with method-table
    # lookups — identical for every site with the same callee/arity,
    # so cache per pass invocation. (Lowered sites are erased; the
    # fixpoint re-enters this pass only for still-unresolved ones.)
    arg_keyed_cache = Dict{Tuple{Any, Int, Any}, Vector{Tuple{DataType, Any}}}()
    arity_cands_cache = Dict{Int, Vector{Core.MethodInstance}}()
    ctor_cands_cache = Dict{Int, Vector{Tuple{DataType, Core.MethodInstance}}}()
    for fname in ("jl_apply_generic", "ijl_apply_generic")
        haskey(LLVM.functions(mod), fname) || continue
        agfn = LLVM.functions(mod)[fname]
        for use in collect(LLVM.uses(agfn))
            call = LLVM.user(use)
            call isa LLVM.CallInst || continue
            args = collect(LLVM.arguments(call))
            dbg = haskey(ENV, "WHISK_DEBUG_DISPATCH")
            if length(args) < 3
                dbg && @info "apply_generic site skipped: $(length(args)) args" call=string(call)
                continue
            end
            fval, frame, nargs_v = args[1], args[2], args[3]
            if !(nargs_v isa LLVM.ConstantInt)
                dbg && @info "apply_generic site skipped: non-constant nargs" call=string(call)
                continue
            end
            nargs = convert(Int, nargs_v)

            # CONSTANT-SITE DEVIRTUALIZATION (what const-prop does
            # natively): if the callee and every frame argument decode
            # to host constants, the exact method is known — emit a
            # direct call, no switch. Covers the common `T[]` /
            # `getindex(::Type)` literal pattern whose callee
            # (`getindex`) has too many arity-matches for type-keyed
            # dispatch. Frame slot values are recovered from the
            # `store`s preceding the call.
            devirt_done = false
            fobj_known = decode_host_object(fval)
            fobj_const = fobj_known !== nothing &&
                ((fobj_known isa Function && Base.issingletontype(typeof(fobj_known))) ||
                 fobj_known isa Type)
            dbg && fobj_known === nothing &&
                @info "apply_generic site: callee constant undecodable" nargs fn=LLVM.name(LLVM.parent(LLVM.parent(call))) fval=first(string(fval), 200)
            slot_vals = fobj_const ? decode_frame_slots(call, frame, nargs) :
                        nothing
            if fobj_const
                if all(v -> v !== missing && v !== nothing, slot_vals)
                    ats = Any[v isa Type ? Type{v} : typeof(v) for v in slot_vals]
                    ft = fobj_known isa Type ? Type{fobj_known} : typeof(fobj_known)
                    sig = Tuple{ft, ats...}
                    matched = Base._which(sig; raise = false)
                    if matched !== nothing
                        mi = Core.Compiler.specialize_method(matched)
                        sub = _sub_compile_mi!(mod, target, params, mi)
                        if sub !== nothing
                            sub_ft = LLVM.function_type(sub)
                            rtv = LLVM.value_type(call)
                            emitted = false
                            @dispose b = LLVM.IRBuilder() begin
                                LLVM.position!(b, call)
                                LLVM.debuglocation!(b, call)
                                if LLVM.debuglocation(b) === nothing
                                    spc = LLVM.get_subprogram(LLVM.parent(LLVM.parent(call)))
                                    spc !== nothing &&
                                        LLVM.debuglocation!(b, LLVM.DILocation(0, 0, spc))
                                end
                                loaded = dispatch_call_args!(b, sub, mi, fval, frame, i64)
                                if loaded !== nothing
                                    r = LLVM.call!(b, sub_ft, sub, loaded)
                                    LLVM.return_type(sub_ft) == rtv &&
                                        LLVM.replace_uses!(call, r)
                                    emitted = true
                                end
                            end
                            if emitted
                                rtv != LLVM.VoidType() && !isempty(LLVM.uses(call)) &&
                                    LLVM.replace_uses!(call, LLVM.UndefValue(rtv))
                                LLVM.erase!(call)
                                n += 1
                                devirt_done = true
                            elseif dbg
                                @info "apply_generic devirt: signature/ABI mismatch, falling through" mi=string(mi)
                            end
                        end
                    end
                end
            end
            devirt_done && continue

            # PARTIAL DEVIRTUALIZATION (known-singleton/Type callee,
            # some runtime args): frame slots that decoded to host
            # constants become FIXED signature types — `convert(Float64,
            # v)` fixes slot 1 to the ghost `Type{Float64}` the sub-fn
            # never receives — and only the RUNTIME slots key the
            # dispatch switch: outer tag switch on the first runtime
            # slot, nested switch on the second (`+(x, v)` — every `+`
            # method constrains BOTH operands). The all-runtime case
            # degenerates to the classic first-argument table
            # (`observable[]` inside @nospecialize'd code).
            arg_keyed = Tuple{DataType, Any}[]   # (U, mi) | (U, [(V, mi)...])
            runtime_slots = collect(1:nargs)
            if fobj_const && nargs >= 1
                fixed = Any[v === missing || v === nothing ? missing : v
                            for v in slot_vals]
                runtime_slots = [i for i in 1:nargs if fixed[i] === missing]
                fk_type = fobj_known isa Type ? Type{fobj_known} : typeof(fobj_known)
                if !isempty(runtime_slots)
                    fixed_key = ntuple(i -> fixed[i] === missing ? nothing :
                                            (fixed[i] isa Type ? Type{fixed[i]} :
                                             typeof(fixed[i])), nargs)
                    arg_keyed = get!(arg_keyed_cache, (fk_type, nargs, fixed_key)) do
                        # Dynsite provenance bounds each slot's
                        # candidate sweep; without it (runtime-helper
                        # code outside the walked world) the budget cap
                        # keeps a method-rich callee from scheduling
                        # hundreds of sub-compiles.
                        hints = dynsite_hints(world, fk_type, nargs)
                        arg_keyed_candidates(fk_type, nargs, world.universe;
                                             hints, fixed,
                                             budget = hints === nothing ? 64 : 256)
                    end
                end
                dbg && isempty(arg_keyed) &&
                    @info "apply_generic site: known callee but empty arg-keyed table" callee=fobj_known nargs fn=LLVM.name(LLVM.parent(LLVM.parent(call))) slots=join([v === missing ? "?" : repr(v) for v in slot_vals], " | ")
            end

            # Candidates: every reachable Function type with a method
            # for `nargs` arguments. We match on arity here — the
            # per-site argtypes from `world.dynsites` refine this once
            # callsites carry provenance; arity + objectid identity is
            # already sound (a wrong-arity candidate's objectid can't
            # appear in the tag slot of the callee object).
            cands = get!(arity_cands_cache, nargs) do
                acc = Core.MethodInstance[]
                for T in world.universe
                    T <: Function || continue
                    T <: Core.Builtin && continue
                    T === Core.IntrinsicFunction && continue
                    # Enumerate T's call methods and keep arity matches.
                    # `Tuple{T, Any...}` would NOT subtype a typed closure
                    # signature like `Tuple{T, Float64}`, so a `_which`
                    # lookup misses — `_methods_by_ftype` with Vararg
                    # enumerates instead, and the match's intersected
                    # spec_types carries the method's own argument types
                    # for specialization.
                    matches = Base._methods_by_ftype(Tuple{T, Vararg{Any}}, -1,
                                                      Base.get_world_counter())
                    matches isa Vector || continue   # `nothing` = lookup overflow
                    t_matches = Core.MethodInstance[]
                    for m in matches
                        m.spec_types isa DataType || continue  # UnionAll match → non-concrete
                        ps = m.spec_types.parameters
                        # `nargs` is the C-convention count: frame args
                        # EXCLUDING the callee. Method signatures include
                        # the callee type as parameter 1.
                        length(ps) == nargs + 1 || continue
                        # Callee type must be concrete; arg params may be
                        # `Any` (the arg stays boxed — untyped closure
                        # params like `do v; …` — interior dynamics
                        # resolve in the fixpoint).
                        (ps[1] isa DataType && isconcretetype(ps[1])) || continue
                        all(p -> p === Any || (p isa DataType && isconcretetype(p)),
                            ps[2:end]) || continue
                        push!(t_matches, Core.Compiler.specialize_method(m))
                    end
                    # One switch case per callee TYPE: with several
                    # arity-matching methods the case would need argtype
                    # discrimination (v2) — skip the type, the default
                    # trap names it.
                    if length(t_matches) == 1
                        push!(acc, t_matches[1])
                    elseif length(t_matches) > 1 && haskey(ENV, "WHISK_DEBUG_DISPATCH")
                        @info "apply_generic: $(T) has $(length(t_matches)) arity-$(nargs) methods — needs argtype-refined dispatch (v2); skipped"
                    end
                end
                acc
            end
            # Constructor candidates: callees that are TYPE objects
            # (`Pair{Int,Any}(…)`). These dispatch by POINTER IDENTITY
            # of the callee against the type descriptor global —
            # descriptors carry no tag prefix, so the oid path cannot
            # classify them.
            ctor_cands = get!(ctor_cands_cache, nargs) do
                acc = Tuple{DataType, Core.MethodInstance}[]
                for X in world.universe
                    ctor_matches = Base._methods_by_ftype(Tuple{Type{X}, Vararg{Any}}, -1,
                                                           Base.get_world_counter())
                    ctor_matches isa Vector || continue   # `nothing` = lookup overflow
                    xm = Core.MethodInstance[]
                    for m in ctor_matches
                        m.spec_types isa DataType || continue
                        ps = m.spec_types.parameters
                        length(ps) == nargs + 1 || continue
                        all(p -> p === Any || (p isa DataType && isconcretetype(p)),
                            ps[2:end]) || continue
                        push!(xm, Core.Compiler.specialize_method(m))
                    end
                    length(xm) == 1 && push!(acc, (X, xm[1]))
                end
                acc
            end

            # Dedup across sources by callee type (one case per type).
            bytype = Dict{Any, Core.MethodInstance}()
            for mi in cands
                st = mi.specTypes
                st isa DataType || continue
                bytype[st.parameters[1]] = mi
            end
            cands = collect(values(bytype))
            if isempty(cands) && isempty(ctor_cands) && isempty(arg_keyed)
                dbg && @info "apply_generic site: no candidates (nargs=$(nargs)) among $(count(T -> T <: Function, world.universe)) function types"
                continue
            end

            compiled = Tuple{DataType, LLVM.Function, Core.MethodInstance}[]
            for mi in cands
                T = mi.specTypes.parameters[1]
                T isa DataType || continue
                sub = _sub_compile_mi!(mod, target, params, mi)
                sub === nothing && continue
                push!(compiled, (T, sub, mi))
            end
            ctor_compiled = Tuple{DataType, LLVM.Function, Core.MethodInstance}[]
            for (X, mi) in ctor_cands
                sub = _sub_compile_mi!(mod, target, params, mi)
                sub === nothing && continue
                push!(ctor_compiled, (X, sub, mi))
            end
            # Arg-keyed mode replaces callee-keyed dispatch entirely
            # when the callee is a known singleton: the switch then
            # scrutinises frame[0]'s tag. Entries refined on the second
            # argument compile to a NESTED payload — a per-(U, V) sub
            # list the emitter turns into an inner tag switch on
            # frame[1].
            arg_keyed_mode = !isempty(arg_keyed)
            refined_compiled = Dict{DataType, Vector{Tuple{DataType, LLVM.Function, Core.MethodInstance}}}()
            if arg_keyed_mode
                dbg && @info "apply_generic arg-keyed site: callee=$(fobj_known) nargs=$nargs " *
                    "cases=$(length(arg_keyed)) sub-compiles=$(sum(p -> p[2] isa Core.MethodInstance ? 1 : length(p[2]), arg_keyed; init = 0))"
                compiled = Tuple{DataType, LLVM.Function, Core.MethodInstance}[]
                for (U, payload) in arg_keyed
                    if payload isa Core.MethodInstance
                        sub = _sub_compile_mi!(mod, target, params, payload)
                        sub === nothing && continue
                        push!(compiled, (U, sub, payload))
                    else
                        inner = Tuple{DataType, LLVM.Function, Core.MethodInstance}[]
                        for (V, mi) in payload
                            sub = _sub_compile_mi!(mod, target, params, mi)
                            sub === nothing && continue
                            push!(inner, (V, sub, mi))
                        end
                        isempty(inner) && continue
                        refined_compiled[U] = inner
                        # The outer switch case exists; its body is the
                        # inner switch (sub/mi unused for these).
                        push!(compiled, (U, first(inner)[2], first(inner)[3]))
                    end
                end
                empty!(ctor_compiled)
            end
            if isempty(compiled) && isempty(ctor_compiled)
                dbg && @info "apply_generic site: all $(length(cands)) candidate sub-compiles failed"
                continue
            end
            dbg && @info "apply_generic site resolved" candidates=[string(T) for (T, _) in compiled] ctors=[string(X) for (X, _) in ctor_compiled]

            bb = LLVM.parent(call)
            fn = LLVM.parent(bb)
            tail_bb = LLVM.BasicBlock(fn, "ag_tail")
            default_bb = LLVM.BasicBlock(fn, "ag_default")
            case_bbs = [LLVM.BasicBlock(fn, "ag_case_$(i)") for i in 1:length(compiled)]
            ctor_bbs = [LLVM.BasicBlock(fn, "ag_ctor_$(i)") for i in 1:length(ctor_compiled)]
            ctor_chk = [LLVM.BasicBlock(fn, "ag_ctorchk_$(i)") for i in 2:length(ctor_compiled)]
            oid_bb = isempty(compiled) ? default_bb : LLVM.BasicBlock(fn, "ag_oid")

            # Move everything after `call` into tail_bb.
            insts_after = LLVM.Instruction[]
            seen = false
            for inst in collect(LLVM.instructions(bb))
                seen && push!(insts_after, inst)
                inst === call && (seen = true)
            end
            @dispose bmove = LLVM.IRBuilder() begin
                LLVM.position!(bmove, tail_bb)
                for inst in insts_after
                    LLVM.remove!(inst)
                    LLVM.insert!(bmove, inst)
                end
            end
            # The terminator moved to tail_bb: successor PHIs that
            # listed `bb` as the incoming block must now list
            # `tail_bb`. The C API has no set-incoming, so rebuild
            # each affected phi with corrected incoming and RAUW.
            let term = LLVM.terminator(tail_bb)
                term === nothing && error("Whisk: split block lost its terminator")
                for succ in LLVM.successors(term)
                    for inst in collect(LLVM.instructions(succ))
                        inst isa LLVM.PHIInst || break
                        inc = collect(LLVM.incoming(inst))
                        any(p -> p[2] == bb, inc) || continue
                        @dispose bp = LLVM.IRBuilder() begin
                            LLVM.position!(bp, inst)
                            newphi = LLVM.phi!(bp, LLVM.value_type(inst))
                            LLVM.append!(LLVM.incoming(newphi),
                                Tuple{LLVM.Value, LLVM.BasicBlock}[
                                    (v, blk == bb ? tail_bb : blk)
                                    for (v, blk) in inc])
                            LLVM.replace_uses!(inst, newphi)
                        end
                        LLVM.erase!(inst)
                    end
                end
            end

            rt = LLVM.value_type(call)
            results = Tuple{LLVM.Value, LLVM.BasicBlock}[]
            @dispose b = LLVM.IRBuilder() begin
                LLVM.position!(b, bb)
                # Inherit the original call's !dbg — inlinable calls in
                # debug-info functions must carry a location. If the
                # site has none, synthesize line 0 in the CALLER's
                # subprogram (callee scopes would fail the verifier).
                LLVM.debuglocation!(b, call)
                if LLVM.debuglocation(b) === nothing
                    spc = LLVM.get_subprogram(fn)
                    spc !== nothing &&
                        LLVM.debuglocation!(b, LLVM.DILocation(0, 0, spc))
                end
                # CONSTRUCTOR identity chain first: a Type-object
                # callee IS the descriptor global (descriptors carry no
                # tag prefix, so the oid path below cannot classify
                # them). Falls through to instance dispatch.
                for (i, (X, _)) in enumerate(ctor_compiled)
                    desc = whisk_type_descriptor!(mod, X)
                    is_x = LLVM.icmp!(b, LLVM.API.LLVMIntEQ, fval, desc)
                    next_bb = i < length(ctor_compiled) ? ctor_chk[i] : oid_bb
                    LLVM.br!(b, is_x, ctor_bbs[i], next_bb)
                    i < length(ctor_compiled) && LLVM.position!(b, ctor_chk[i])
                end
                if isempty(ctor_compiled)
                    LLVM.br!(b, oid_bb)
                end
                # tag-oid of a boxed object: `load((load(obj-8) & ~15) + 16)`.
                load_tag_oid! = function (obj)
                    o_int = LLVM.ptrtoint!(b, obj, i64)
                    tag_p = LLVM.inttoptr!(b, LLVM.sub!(b, o_int, LLVM.ConstantInt(i64, 8)),
                                            LLVM.PointerType(i64))
                    tag = LLVM.and!(b, LLVM.load!(b, i64, tag_p),
                                     LLVM.ConstantInt(i64, -16))
                    oid_p = LLVM.inttoptr!(b, LLVM.add!(b, tag, LLVM.ConstantInt(i64, 16)),
                                            LLVM.PointerType(i64))
                    return LLVM.load!(b, i64, oid_p)
                end
                if !isempty(compiled)
                    LLVM.position!(b, oid_bb)
                    scrut_obj = fval
                    if arg_keyed_mode
                        # Key on the FIRST RUNTIME slot (slot 1 unless
                        # partial devirt fixed leading args).
                        r1 = runtime_slots[1]
                        slot_p = r1 == 1 ? frame :
                            LLVM.gep!(b, LLVM.PointerType(LLVM.Int8Type()),
                                      frame, [LLVM.ConstantInt(i64, r1 - 1)])
                        scrut_obj = LLVM.load!(b,
                            LLVM.PointerType(LLVM.Int8Type()), slot_p)
                    end
                    oid = load_tag_oid!(scrut_obj)
                    sw = LLVM.switch!(b, oid, default_bb, length(compiled))
                    for (i, (T, _, _)) in enumerate(compiled)
                        LLVM.API.LLVMAddCase(sw,
                            LLVM.ConstantInt(i64, reinterpret(Int64, UInt64(objectid(T)))),
                            case_bbs[i])
                    end
                end

                emit_case! = function (case_bb, sub, mi)
                    LLVM.position!(b, case_bb)
                    sub_ft = LLVM.function_type(sub)
                    # Signature-driven slot mapping: ghosts (singleton
                    # callee types, `Type{X}` params) occupy frame
                    # slots but have no LLVM parameter.
                    loaded = dispatch_call_args!(b, sub, mi, fval, frame, i64)
                    if loaded === nothing
                        # Un-mappable ABI: keep dispatch sound — this
                        # case traps like the default.
                        LLVM.br!(b, default_bb)
                        return
                    end
                    r = LLVM.call!(b, sub_ft, sub, loaded)
                    res = if LLVM.return_type(sub_ft) == rt
                        r
                    elseif rt isa LLVM.PointerType &&
                           LLVM.return_type(sub_ft) != LLVM.VoidType()
                        # Candidate returns an unboxed scalar but the
                        # apply_generic contract is a boxed ptr — box
                        # via the runtime helper. Unboxable returns
                        # keep undef (uses, if any, trap downstream).
                        boxed = box_primitive_result!(b, mod, mi, r)
                        boxed === nothing ? LLVM.UndefValue(rt) : boxed
                    else
                        LLVM.UndefValue(rt)
                    end
                    push!(results, (res, case_bb))
                    LLVM.br!(b, tail_bb)
                end
                for (i, (T, sub, mi)) in enumerate(compiled)
                    inner = get(refined_compiled, T, nothing)
                    if inner === nothing
                        emit_case!(case_bbs[i], sub, mi)
                    else
                        # Second-runtime-slot refinement: inner tag
                        # switch over the (V, mi) pairs.
                        LLVM.position!(b, case_bbs[i])
                        r2 = runtime_slots[2]
                        slot2 = LLVM.gep!(b, LLVM.PointerType(LLVM.Int8Type()),
                                          frame, [LLVM.ConstantInt(i64, r2 - 1)])
                        arg2 = LLVM.load!(b, LLVM.PointerType(LLVM.Int8Type()), slot2)
                        oid2 = load_tag_oid!(arg2)
                        vbbs = [LLVM.BasicBlock(fn, "ag_case_$(i)_v$(j)")
                                for j in 1:length(inner)]
                        sw2 = LLVM.switch!(b, oid2, default_bb, length(inner))
                        for (j, (V, _, _)) in enumerate(inner)
                            LLVM.API.LLVMAddCase(sw2,
                                LLVM.ConstantInt(i64, reinterpret(Int64, UInt64(objectid(V)))),
                                vbbs[j])
                        end
                        for (j, (V, vsub, vmi)) in enumerate(inner)
                            emit_case!(vbbs[j], vsub, vmi)
                        end
                    end
                end
                for (i, (X, sub, mi)) in enumerate(ctor_compiled)
                    emit_case!(ctor_bbs[i], sub, mi)
                end

                LLVM.position!(b, default_bb)
                if haskey(ENV, "WHISK_DEBUG_DISPATCH")
                    # Debug import: report the runtime scrutinee oid +
                    # callee pointer at the failed dispatch.
                    dbg_fty = LLVM.FunctionType(LLVM.VoidType(), [i64, i64])
                    dbg_fn = haskey(LLVM.functions(mod), "whisk_debug_dispatch") ?
                             LLVM.functions(mod)["whisk_debug_dispatch"] :
                             LLVM.Function(mod, "whisk_debug_dispatch", dbg_fty)
                    df_int = LLVM.ptrtoint!(b, fval, i64)
                    dtag_p = LLVM.inttoptr!(b,
                        LLVM.sub!(b, df_int, LLVM.ConstantInt(i64, 8)),
                        LLVM.PointerType(i64))
                    dtag = LLVM.and!(b, LLVM.load!(b, i64, dtag_p),
                                      LLVM.ConstantInt(i64, -16))
                    doid_p = LLVM.inttoptr!(b,
                        LLVM.add!(b, dtag, LLVM.ConstantInt(i64, 16)),
                        LLVM.PointerType(i64))
                    doid = LLVM.load!(b, i64, doid_p)
                    LLVM.call!(b, dbg_fty, dbg_fn, LLVM.Value[doid, df_int])
                end
                msg = (_dbg_location_string(call) == "" ? "?:?" :
                       _dbg_location_string(call)) *
                      " [in " * LLVM.name(fn) * ", nargs=" * string(nargs) * "]" *
                      ": dynamic dispatch — callee not in the closed world (" *
                      join(string.(first.(compiled)), ", ") * ")"
                js_fn = _get_or_declare_js_console!(mod)
                trap_fn = _get_or_declare_trap!(mod)
                gvm = _intern_message_global!(mod, msg)
                LLVM.call!(b, LLVM.function_type(js_fn), js_fn, LLVM.Value[
                    LLVM.ConstantInt(LLVM.Int32Type(), 2), gvm,
                    LLVM.ConstantInt(i64, sizeof(msg))])
                LLVM.call!(b, LLVM.function_type(trap_fn), trap_fn, LLVM.Value[])
                LLVM.unreachable!(b)

                # Result phi in tail.
                LLVM.position!(b, tail_bb)
                if rt != LLVM.VoidType() && !isempty(LLVM.uses(call))
                    LLVM.position!(b, first(LLVM.instructions(tail_bb)))
                    phi = LLVM.phi!(b, rt)
                    LLVM.append!(LLVM.incoming(phi), results)
                    LLVM.replace_uses!(call, phi)
                end
            end
            LLVM.erase!(call)
            n += 1
        end
    end
    return n
end

"""
    lower_subtype_checks!(mod, world) → Int

Closed-world `jl_subtype` lowering. `isa`/typeassert against an
ABSTRACT type can't be an exact tag compare, so codegen calls the
runtime: `i32 ijl_subtype(ptr a, ptr b)` with `a` = the value's type
(tag) and `b` = the abstract type constant. With a closed world the
predicate is decidable per concrete type: replace the call with an
objectid-switch over the universe members that satisfy `<: B`
(returns 1), default 0. `b` must peel to a `_whisk_type_*` descriptor
global (the reverse registry recovers the Julia type); sites with an
unrecognisable RHS are left for the import-validator to flag.
"""
function lower_subtype_checks!(mod::LLVM.Module, world)
    n = 0
    i64 = LLVM.Int64Type()
    for fname in ("jl_subtype", "ijl_subtype")
        haskey(LLVM.functions(mod), fname) || continue
        fn = LLVM.functions(mod)[fname]
        rt = LLVM.return_type(LLVM.function_type(fn))
        for use in collect(LLVM.uses(fn))
            call = LLVM.user(use)
            call isa LLVM.CallInst || continue
            args = collect(LLVM.arguments(call))
            length(args) == 2 || continue
            # The RHS may be a direct descriptor constant, or a load
            # through a `@jl_global_N` slot whose initializer is the
            # descriptor (the host-pointer rewrite fixed initializers).
            barg = args[2]
            if barg isa LLVM.LoadInst
                src = LLVM.operands(barg)[1]
                if src isa LLVM.GlobalVariable && !LLVM.isdeclaration(src)
                    init = LLVM.initializer(src)
                    init === nothing || (barg = init)
                end
            end
            bop = _peel_to_global(barg)
            bop === nothing && continue
            B = get(DESCRIPTOR_NAME_TO_TYPE, LLVM.name(bop), nothing)
            B === nothing && continue
            yes = UInt64[UInt64(objectid(T)) for T in world.universe if T <: B]
            haskey(ENV, "WHISK_DEBUG_DISPATCH") &&
                @info "subtype site" B yes_types=[string(T) for T in world.universe if T <: B]

            bb = LLVM.parent(call)
            fnc = LLVM.parent(bb)
            tail_bb = LLVM.BasicBlock(fnc, "st_tail")
            insts_after = LLVM.Instruction[]
            seen = false
            for inst in collect(LLVM.instructions(bb))
                seen && push!(insts_after, inst)
                inst === call && (seen = true)
            end
            @dispose bmove = LLVM.IRBuilder() begin
                LLVM.position!(bmove, tail_bb)
                for inst in insts_after
                    LLVM.remove!(inst)
                    LLVM.insert!(bmove, inst)
                end
            end
            let term = LLVM.terminator(tail_bb)
                for succ in LLVM.successors(term)
                    for inst in collect(LLVM.instructions(succ))
                        inst isa LLVM.PHIInst || break
                        inc = collect(LLVM.incoming(inst))
                        any(p -> p[2] == bb, inc) || continue
                        @dispose bp = LLVM.IRBuilder() begin
                            LLVM.position!(bp, inst)
                            newphi = LLVM.phi!(bp, LLVM.value_type(inst))
                            LLVM.append!(LLVM.incoming(newphi),
                                Tuple{LLVM.Value, LLVM.BasicBlock}[
                                    (v, blk == bb ? tail_bb : blk)
                                    for (v, blk) in inc])
                            LLVM.replace_uses!(inst, newphi)
                        end
                        LLVM.erase!(inst)
                    end
                end
            end

            hit_bb = LLVM.BasicBlock(fnc, "st_hit")
            miss_bb = LLVM.BasicBlock(fnc, "st_miss")
            @dispose b = LLVM.IRBuilder() begin
                LLVM.position!(b, bb)
                LLVM.debuglocation!(b, call)
                a_int = LLVM.ptrtoint!(b, args[1], i64)
                oid_p = LLVM.inttoptr!(b, LLVM.add!(b, a_int, LLVM.ConstantInt(i64, 16)),
                                        LLVM.PointerType(i64))
                oid = LLVM.load!(b, i64, oid_p)
                sw = LLVM.switch!(b, oid, miss_bb, length(yes))
                for o in yes
                    LLVM.API.LLVMAddCase(sw,
                        LLVM.ConstantInt(i64, reinterpret(Int64, o)), hit_bb)
                end
                LLVM.position!(b, hit_bb); LLVM.br!(b, tail_bb)
                LLVM.position!(b, miss_bb); LLVM.br!(b, tail_bb)
                LLVM.position!(b, first(LLVM.instructions(tail_bb)))
                phi = LLVM.phi!(b, rt)
                LLVM.append!(LLVM.incoming(phi),
                    Tuple{LLVM.Value, LLVM.BasicBlock}[
                        (LLVM.ConstantInt(rt, 1), hit_bb),
                        (LLVM.ConstantInt(rt, 0), miss_bb)])
                LLVM.replace_uses!(call, phi)
            end
            LLVM.erase!(call)
            n += 1
        end
    end
    return n
end

# Decode a `julia.call` operand to the host object it references —
# constants (`inttoptr`) or loads through `@jl_global` slots. Shared
# shape with `_decode_method_instance`, returning any object.
function decode_host_object(operand)
    # A descriptor global IS the wasm-side identity of its DataType —
    # `@_Core_Float64_NN = global ptr @_whisk_type_<oid>` slots load the
    # type OBJECT, so peeling to the descriptor recovers the type.
    if operand isa LLVM.GlobalVariable
        T = descriptor_global_type(LLVM.name(operand))
        T !== nothing && return T
        init = LLVM.initializer(operand)
        init === nothing && return nothing
        init === operand && return nothing
        return decode_host_object(init)
    end
    if operand isa LLVM.LoadInst
        ptr_op = LLVM.operands(operand)[1]
        ptr_op isa LLVM.GlobalVariable || return nothing
        init = LLVM.initializer(ptr_op)
        init === nothing && return nothing
        return decode_host_object(init)
    end
    if operand isa LLVM.Instruction || operand isa LLVM.ConstantExpr
        oc = LLVM.opcode(operand)
        if oc == LLVM.API.LLVMAddrSpaceCast || oc == LLVM.API.LLVMBitCast ||
           oc == LLVM.API.LLVMIntToPtr || oc == LLVM.API.LLVMPtrToInt
            return decode_host_object(LLVM.operands(operand)[1])
        end
    end
    if operand isa LLVM.ConstantInt
        addr = convert(Int, operand)
        addr <= 0x100000 && return nothing
        addr % 8 == 0 || return nothing
        return try
            Base.unsafe_pointer_to_objref(Ptr{Nothing}(addr))
        catch
            nothing
        end
    end
    return nothing
end

"""
    lower_julia_call_builtins!(mod, world) → Int

Lower `julia.call(ptr @jl_f_<builtin>, ptr task, args…)` wrappers:

  * `jl_f_throw_methoderror` → `js_console` + trap (genuine runtime
    error path, source-located).
  * `jl_f_finalizer` → erased. Whisk's GC has no finalizer support
    (v1); registration is a no-op. Semantic deviation is documented —
    finalizers never run in wasm builds.
  * `jl_f_invokelatest` → rebuilt as `ijl_apply_generic(f, frame, n)`
    so closed-world dispatch resolves it (`invokelatest(f, args…)` ≡
    `f(args…)` in a closed world — there is no later world).
  * `jl_f_getfield(obj, ::Symbol)` with a decodable constant symbol →
    objectid-switch over universe types owning that field, each case
    calling `ijl_get_nth_field_checked(obj, idx)` with the resolved
    constant index; default traps loudly.

Must run inside the fixpoint section (operands still carry host
pointers) and before `lower_apply_generic!`.
"""
function lower_julia_call_builtins!(mod::LLVM.Module, world, target_=nothing, params_=nothing)
    i64 = LLVM.Int64Type()
    i32t = LLVM.Int32Type()
    ptrty = LLVM.PointerType(LLVM.Int8Type())
    n = 0
    # Direct jlcall form: `jl_f_invokelatest(ptr task, ptr frame, i32 n)`
    # with the CALLEE in frame[0]. Rewrite to
    # `ijl_apply_generic(frame[0], frame+8, n-1)` — closed-world
    # dispatch then resolves it (no "later world" exists AOT).
    if haskey(LLVM.functions(mod), "jl_f_invokelatest")
        il = LLVM.functions(mod)["jl_f_invokelatest"]
        invoke_fty = LLVM.FunctionType(ptrty, [ptrty, ptrty, i32t])
        ag = haskey(LLVM.functions(mod), "ijl_apply_generic") ?
             LLVM.functions(mod)["ijl_apply_generic"] :
             LLVM.Function(mod, "ijl_apply_generic", invoke_fty)
        for use in collect(LLVM.uses(il))
            call = LLVM.user(use)
            call isa LLVM.CallInst || continue
            args = collect(LLVM.arguments(call))
            length(args) == 3 || continue
            frame, nargs_v = args[2], args[3]
            nargs_v isa LLVM.ConstantInt || continue
            nargs = convert(Int, nargs_v)
            nargs >= 1 || continue
            @dispose b = LLVM.IRBuilder() begin
                LLVM.position!(b, call)
                LLVM.debuglocation!(b, call)
                f = LLVM.load!(b, ptrty, frame)
                rest = LLVM.gep!(b, ptrty, frame, [LLVM.ConstantInt(i64, 1)])
                newc = LLVM.call!(b, invoke_fty, ag, LLVM.Value[
                    f, rest, LLVM.ConstantInt(i32t, nargs - 1)])
                LLVM.replace_uses!(call, newc)
            end
            LLVM.erase!(call)
            n += 1
        end
        isempty(LLVM.uses(il)) && LLVM.erase!(il)
    end
    haskey(LLVM.functions(mod), "julia.call") || return n
    callfn = LLVM.functions(mod)["julia.call"]
    for use in collect(LLVM.uses(callfn))
        call = LLVM.user(use)
        call isa LLVM.CallInst || continue
        args = collect(LLVM.arguments(call))
        length(args) >= 2 || continue
        target = args[1]
        target isa LLVM.Function || continue
        tname = LLVM.name(target)

        if tname == "jl_f_throw_methoderror"
            js_fn = _get_or_declare_js_console!(mod)
            trap_fn = _get_or_declare_trap!(mod)
            loc = _dbg_location_string(call)
            msg = (isempty(loc) ? "?:?" : loc) * ": MethodError"
            gvm = _intern_message_global!(mod, msg)
            @dispose b = LLVM.IRBuilder() begin
                LLVM.position!(b, call)
                LLVM.debuglocation!(b, call)
                LLVM.call!(b, LLVM.function_type(js_fn), js_fn, LLVM.Value[
                    LLVM.ConstantInt(i32t, 2), gvm,
                    LLVM.ConstantInt(i64, sizeof(msg))])
                LLVM.call!(b, LLVM.function_type(trap_fn), trap_fn, LLVM.Value[])
            end
            rt = LLVM.value_type(call)
            rt != LLVM.VoidType() && LLVM.replace_uses!(call, LLVM.UndefValue(rt))
            LLVM.erase!(call)
            n += 1
        elseif tname == "jl_f_finalizer"
            rt = LLVM.value_type(call)
            rt != LLVM.VoidType() && LLVM.replace_uses!(call, LLVM.null(rt))
            LLVM.erase!(call)
            n += 1
        elseif (tname == "jl_apply_generic" || tname == "ijl_apply_generic") &&
               length(args) >= 2
            # `julia.call(@jl_apply_generic, F, args…)`: unlike the
            # jl_f_* builtins (whose slot 2 holds the BUILTIN's own
            # function object and whose real args start at slot 3),
            # apply_generic's F IS the dispatched callee — `args[2]`.
            # Reading the callee from slot 3 dropped `F` and promoted
            # the first argument to callee (`x + v` dispatched on the
            # boxed `x` with frame `[v]`).
            f = args[2]
            rest = args[3:end]
            # Devirtualize when callee + args decode to host constants
            # (what const-prop does natively): a singleton Function
            # callee with concretely-typed args names one method.
            fobj = decode_host_object(f)
            if fobj isa Function && Base.issingletontype(typeof(fobj)) &&
               target_ !== nothing
                ats = Any[]
                ok = true
                for a in rest
                    o = decode_host_object(a)
                    if o === nothing
                        ok = false; break
                    end
                    push!(ats, typeof(o))
                end
                if ok && all(t -> t isa DataType && isconcretetype(t), ats)
                    sig = Tuple{typeof(fobj), ats...}
                    matched = Base._which(sig; raise = false)
                    if matched !== nothing
                        mi = Core.Compiler.specialize_method(matched)
                        sub = _sub_compile_mi!(mod, target_, params_, mi)
                        if sub !== nothing
                            sub_ft = LLVM.function_type(sub)
                            sub_ps = LLVM.parameters(sub_ft)
                            @dispose b = LLVM.IRBuilder() begin
                                LLVM.position!(b, call)
                                LLVM.debuglocation!(b, call)
                                loaded = LLVM.Value[]
                                srcargs = LLVM.Value[f, rest...]
                                nfixed = min(length(sub_ps), length(srcargs))
                                for j in 1:length(sub_ps)
                                    pty = sub_ps[j]
                                    a = srcargs[min(j, length(srcargs))]
                                    if pty isa LLVM.PointerType
                                        push!(loaded, a)
                                    else
                                        push!(loaded, LLVM.load!(b, pty, a))
                                    end
                                end
                                r = LLVM.call!(b, sub_ft, sub, loaded)
                                LLVM.value_type(call) == LLVM.return_type(sub_ft) &&
                                    LLVM.replace_uses!(call, r)
                            end
                            rt2 = LLVM.value_type(call)
                            rt2 != LLVM.VoidType() && !isempty(LLVM.uses(call)) &&
                                LLVM.replace_uses!(call, LLVM.UndefValue(rt2))
                            LLVM.erase!(call)
                            n += 1
                            continue
                        end
                    end
                end
            end
            # Fallback: rebuild as the direct `ijl_apply_generic(f,
            # frame, n)` shape — the next fixpoint iteration resolves
            # it via closed-world dispatch.
            invoke_fty = LLVM.FunctionType(ptrty, [ptrty, ptrty, i32t])
            ag = haskey(LLVM.functions(mod), "ijl_apply_generic") ?
                 LLVM.functions(mod)["ijl_apply_generic"] :
                 LLVM.Function(mod, "ijl_apply_generic", invoke_fty)
            @dispose b = LLVM.IRBuilder() begin
                LLVM.position!(b, call)
                LLVM.debuglocation!(b, call)
                frame_ty = LLVM.ArrayType(ptrty, max(length(rest), 1))
                frame = LLVM.alloca!(b, frame_ty)
                for (i, a) in enumerate(rest)
                    slot = LLVM.gep!(b, frame_ty, frame,
                        [LLVM.ConstantInt(i64, 0), LLVM.ConstantInt(i64, i - 1)])
                    LLVM.store!(b, a, slot)
                end
                newc = LLVM.call!(b, invoke_fty, ag, LLVM.Value[
                    f, frame, LLVM.ConstantInt(i32t, length(rest))])
                LLVM.replace_uses!(call, newc)
            end
            LLVM.erase!(call)
            n += 1
        elseif tname == "jl_f_invokelatest" && length(args) >= 3
            f = args[3]
            rest = args[4:end]
            invoke_fty = LLVM.FunctionType(ptrty, [ptrty, ptrty, i32t])
            ag = haskey(LLVM.functions(mod), "ijl_apply_generic") ?
                 LLVM.functions(mod)["ijl_apply_generic"] :
                 LLVM.Function(mod, "ijl_apply_generic", invoke_fty)
            @dispose b = LLVM.IRBuilder() begin
                LLVM.position!(b, call)
                LLVM.debuglocation!(b, call)
                frame_ty = LLVM.ArrayType(ptrty, length(rest))
                frame = LLVM.alloca!(b, frame_ty)
                for (i, a) in enumerate(rest)
                    slot = LLVM.gep!(b, frame_ty, frame,
                        [LLVM.ConstantInt(i64, 0), LLVM.ConstantInt(i64, i - 1)])
                    LLVM.store!(b, a, slot)
                end
                newc = LLVM.call!(b, invoke_fty, ag, LLVM.Value[
                    f, frame, LLVM.ConstantInt(i32t, length(rest))])
                LLVM.replace_uses!(call, newc)
            end
            LLVM.erase!(call)
            n += 1
        elseif tname == "jl_f_tuple" && target_ !== nothing
            # Generic boxed-tuple construction. The semantics live in
            # JULIA (`whisk_rt_tuple(args...) = inferencebarrier(args)`,
            # runtime/intrinsics.jl) — sub-compiled here for the site's
            # arity so allocation, tagging and the boxed return all go
            # through the normal pipeline. Real args start at slot 3
            # (slot 2 is the builtin's F-slot, `ptr null`).
            elems = args[3:end]
            N = length(elems)
            handled = false
            if N == 0
                singleton = host_singleton_constant!(mod, "jl_emptytuple")
                if singleton !== nothing
                    LLVM.replace_uses!(call, singleton)
                    handled = true
                end
            else
                # `whisk_rt_tuple` is Whisk's own helper with exactly
                # one Vararg method — specialize it for this arity
                # directly, no lookup that could miss.
                m = only(methods(whisk_rt_tuple))
                atype = Tuple{typeof(whisk_rt_tuple), ntuple(_ -> Any, N)...}
                mi = Core.Compiler.specialize_method(m, atype, Core.svec())
                sub = _sub_compile_mi!(mod, target_, params_, mi)
                if sub !== nothing
                    sub_ft = LLVM.function_type(sub)
                    sub_ps = LLVM.parameters(sub_ft)
                    if length(sub_ps) == N &&
                       all(p -> p isa LLVM.PointerType, sub_ps) &&
                       LLVM.return_type(sub_ft) isa LLVM.PointerType
                        @dispose b = LLVM.IRBuilder() begin
                            position_with_dbg!(b, call)
                            r = LLVM.call!(b, sub_ft, sub,
                                           LLVM.Value[elems...])
                            LLVM.replace_uses!(call, r)
                        end
                        handled = true
                    elseif haskey(ENV, "WHISK_DEBUG_DISPATCH")
                        @info "jl_f_tuple: whisk_rt_tuple ABI mismatch (N=$(N), params=$(length(sub_ps)))"
                    end
                end
            end
            if !handled
                # No helper available for this arity/ABI: trap loudly
                # rather than leaving an unresolvable import.
                js_fn = _get_or_declare_js_console!(mod)
                trap_fn = _get_or_declare_trap!(mod)
                loc = _dbg_location_string(call)
                msg = (isempty(loc) ? "?:?" : loc) *
                      ": `jl_f_tuple` lowering failed (arity $(N))"
                gvm = _intern_message_global!(mod, msg)
                @dispose b = LLVM.IRBuilder() begin
                    position_with_dbg!(b, call)
                    LLVM.call!(b, LLVM.function_type(js_fn), js_fn, LLVM.Value[
                        LLVM.ConstantInt(i32t, 2), gvm,
                        LLVM.ConstantInt(i64, sizeof(msg))])
                    LLVM.call!(b, LLVM.function_type(trap_fn), trap_fn,
                               LLVM.Value[])
                end
                rt = LLVM.value_type(call)
                rt != LLVM.VoidType() &&
                    LLVM.replace_uses!(call, LLVM.UndefValue(rt))
            end
            LLVM.erase!(call)
            n += 1
        elseif tname in ("jl_f__svec_ref", "jl_f__apply_iterate")
            # SimpleVector indexing / splat-apply need host type-object
            # internals (`T.parameters` data) the descriptor system
            # doesn't carry — static-data-image work (ARCHITECTURE
            # phase 4). They surface on type-printing error paths;
            # trap loudly instead of importing an unresolvable symbol.
            js_fn = _get_or_declare_js_console!(mod)
            trap_fn = _get_or_declare_trap!(mod)
            loc = _dbg_location_string(call)
            msg = (isempty(loc) ? "?:?" : loc) * ": `" * tname *
                  "` (runtime type introspection) is unsupported on wasm"
            gvm = _intern_message_global!(mod, msg)
            @dispose b = LLVM.IRBuilder() begin
                position_with_dbg!(b, call)
                LLVM.call!(b, LLVM.function_type(js_fn), js_fn, LLVM.Value[
                    LLVM.ConstantInt(i32t, 2), gvm,
                    LLVM.ConstantInt(i64, sizeof(msg))])
                LLVM.call!(b, LLVM.function_type(trap_fn), trap_fn, LLVM.Value[])
            end
            rt = LLVM.value_type(call)
            rt != LLVM.VoidType() && LLVM.replace_uses!(call, LLVM.UndefValue(rt))
            LLVM.erase!(call)
            n += 1
        elseif tname == "jl_f_getfield" && length(args) == 4 && world !== nothing
            sym = decode_host_object(args[4])
            if !(sym isa Symbol) && args[4] isa LLVM.LoadInst
                # Julia 1.12 emits interned-symbol slots as EXTERNAL
                # declarations `@jl_sym_<name>_<id>` (JIT-resolved on
                # the host) — the name itself carries the symbol,
                # same naming ABI as the `jlplt_*` thunks.
                src = LLVM.operands(args[4])[1]
                if src isa LLVM.GlobalVariable
                    m = match(r"^jl_sym_(.*)_\d+$", LLVM.name(src))
                    m !== nothing && (sym = Symbol(m.captures[1]))
                end
            end
            if !(sym isa Symbol)
                haskey(ENV, "WHISK_DEBUG_DISPATCH") &&
                    @info "jl_f_getfield: symbol arg undecodable" arg=string(args[4]) got=typeof(sym)
                continue
            end
            # Universe types owning the field, with resolved index.
            owners = Tuple{DataType, Int}[]
            for T in world.universe
                T <: Memory && continue
                isprimitivetype(T) && continue
                idx = findfirst(==(sym), fieldnames(T))
                idx === nothing && continue
                push!(owners, (T, idx))
            end
            if isempty(owners)
                haskey(ENV, "WHISK_DEBUG_DISPATCH") &&
                    @info "jl_f_getfield :$(sym): no owner types in universe"
                continue
            end
            gnf_fty = LLVM.FunctionType(ptrty, [ptrty, i64])
            gnf = haskey(LLVM.functions(mod), "ijl_get_nth_field_checked") ?
                  LLVM.functions(mod)["ijl_get_nth_field_checked"] :
                  LLVM.Function(mod, "ijl_get_nth_field_checked", gnf_fty)
            obj = args[3]
            bb = LLVM.parent(call); fnc = LLVM.parent(bb)
            tail_bb = LLVM.BasicBlock(fnc, "gf_tail")
            default_bb = LLVM.BasicBlock(fnc, "gf_default")
            case_bbs = [LLVM.BasicBlock(fnc, "gf_case_$(i)") for i in 1:length(owners)]
            insts_after = LLVM.Instruction[]; seen = false
            for inst in collect(LLVM.instructions(bb))
                seen && push!(insts_after, inst)
                inst === call && (seen = true)
            end
            @dispose bm = LLVM.IRBuilder() begin
                LLVM.position!(bm, tail_bb)
                for inst in insts_after
                    LLVM.remove!(inst); LLVM.insert!(bm, inst)
                end
            end
            let term = LLVM.terminator(tail_bb)
                for succ in LLVM.successors(term)
                    for inst in collect(LLVM.instructions(succ))
                        inst isa LLVM.PHIInst || break
                        inc = collect(LLVM.incoming(inst))
                        any(p -> p[2] == bb, inc) || continue
                        @dispose bp = LLVM.IRBuilder() begin
                            LLVM.position!(bp, inst)
                            np = LLVM.phi!(bp, LLVM.value_type(inst))
                            LLVM.append!(LLVM.incoming(np),
                                Tuple{LLVM.Value, LLVM.BasicBlock}[
                                    (v, blk == bb ? tail_bb : blk) for (v, blk) in inc])
                            LLVM.replace_uses!(inst, np)
                        end
                        LLVM.erase!(inst)
                    end
                end
            end
            results = LLVM.Value[]
            @dispose b = LLVM.IRBuilder() begin
                LLVM.position!(b, bb)
                LLVM.debuglocation!(b, call)
                if LLVM.debuglocation(b) === nothing
                    spc = LLVM.get_subprogram(fnc)
                    spc !== nothing && LLVM.debuglocation!(b, LLVM.DILocation(0, 0, spc))
                end
                o_int = LLVM.ptrtoint!(b, obj, i64)
                tag_p = LLVM.inttoptr!(b, LLVM.sub!(b, o_int, LLVM.ConstantInt(i64, 8)),
                                        LLVM.PointerType(i64))
                tag = LLVM.and!(b, LLVM.load!(b, i64, tag_p), LLVM.ConstantInt(i64, -16))
                oid_p = LLVM.inttoptr!(b, LLVM.add!(b, tag, LLVM.ConstantInt(i64, 16)),
                                        LLVM.PointerType(i64))
                oid = LLVM.load!(b, i64, oid_p)
                sw = LLVM.switch!(b, oid, default_bb, length(owners))
                for (i, (T, _)) in enumerate(owners)
                    LLVM.API.LLVMAddCase(sw,
                        LLVM.ConstantInt(i64, reinterpret(Int64, UInt64(objectid(T)))),
                        case_bbs[i])
                end
                for (i, (T, idx)) in enumerate(owners)
                    LLVM.position!(b, case_bbs[i])
                    r = LLVM.call!(b, gnf_fty, gnf, LLVM.Value[
                        obj, LLVM.ConstantInt(i64, idx - 1)])
                    push!(results, r)
                    LLVM.br!(b, tail_bb)
                end
                LLVM.position!(b, default_bb)
                js_fn = _get_or_declare_js_console!(mod)
                trap_fn = _get_or_declare_trap!(mod)
                loc = _dbg_location_string(call)
                msg = (isempty(loc) ? "?:?" : loc) *
                      ": getfield :" * string(sym) * " — receiver type not in closed world"
                gvm = _intern_message_global!(mod, msg)
                LLVM.call!(b, LLVM.function_type(js_fn), js_fn, LLVM.Value[
                    LLVM.ConstantInt(i32t, 2), gvm, LLVM.ConstantInt(i64, sizeof(msg))])
                LLVM.call!(b, LLVM.function_type(trap_fn), trap_fn, LLVM.Value[])
                LLVM.unreachable!(b)
                LLVM.position!(b, first(LLVM.instructions(tail_bb)))
                if !isempty(LLVM.uses(call))
                    phi = LLVM.phi!(b, LLVM.value_type(call))
                    LLVM.append!(LLVM.incoming(phi),
                        Tuple{LLVM.Value, LLVM.BasicBlock}[
                            (results[i], case_bbs[i]) for i in 1:length(owners)])
                    LLVM.replace_uses!(call, phi)
                end
            end
            LLVM.erase!(call)
            n += 1
        end
    end
    return n
end

"""
    lower_undecoded_invokes!(mod) → Int

Final fallback for `ijl_invoke` calls whose MethodInstance never
decoded (rare; typically host-only error paths). A source-located
`js_console` + trap replaces the call — the module must never carry a
host-runtime import.
"""
function lower_undecoded_invokes!(mod::LLVM.Module)
    haskey(LLVM.functions(mod), "ijl_invoke") || return 0
    fn = LLVM.functions(mod)["ijl_invoke"]
    js_fn = _get_or_declare_js_console!(mod)
    trap_fn = _get_or_declare_trap!(mod)
    n = 0
    for use in collect(LLVM.uses(fn))
        call = LLVM.user(use)
        call isa LLVM.CallInst || continue
        loc = _dbg_location_string(call)
        msg = (isempty(loc) ? "?:?" : loc) *
              ": invoke target could not be resolved at compile time"
        gvm = _intern_message_global!(mod, msg)
        @dispose b = LLVM.IRBuilder() begin
            LLVM.position!(b, call)
            LLVM.debuglocation!(b, call)
            LLVM.call!(b, LLVM.function_type(js_fn), js_fn, LLVM.Value[
                LLVM.ConstantInt(LLVM.Int32Type(), 2), gvm,
                LLVM.ConstantInt(LLVM.Int64Type(), sizeof(msg))])
            LLVM.call!(b, LLVM.function_type(trap_fn), trap_fn, LLVM.Value[])
        end
        rt = LLVM.value_type(call)
        rt != LLVM.VoidType() && LLVM.replace_uses!(call, LLVM.UndefValue(rt))
        LLVM.erase!(call)
        n += 1
    end
    isempty(LLVM.uses(fn)) && LLVM.erase!(fn)
    return n
end
