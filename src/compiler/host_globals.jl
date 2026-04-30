## Host-VM global scrubbing
##
## After `link_compiler_rt!` brings in `ijl_box_*` and friends, two
## flavours of host-VM global references can survive into the linked
## module — both fatal at `wasm-ld` time:
##
##  1. **Internal globals with a host-pointer initializer**:
##     `@jl_global_NNNN = internal global ptr inttoptr (i64 ADDR to ptr)`.
##     These come from Julia's frontend baking in pointers to host VM
##     `Type{T}` / `DataType` objects for type-check material.
##
##  2. **External declarations of `jl_*` symbols**:
##     `@jl_small_typeof = external local_unnamed_addr constant i8`,
##     `@jl_nothing      = external constant ptr`,
##     `@jl_int64_type`, etc. — declarations that wasm-ld would try to
##     resolve against a host shared library.
##
## We can't usefully translate these to wasm-side data without a real
## type-tag system (that's the v2 dynamic-dispatch story). For now,
## neuter them: rewrite host-pointer initializers to `null`, and turn
## external `jl_*` declarations into internal globals with a zero
## initializer so wasm-ld stops complaining. The runtime never reads
## these values — they only exist as type-check residue that
## optimization should have dropped when types are concrete.

const _whisk_wasm_host_imports = Set([
    "dom_get_element_by_id",
    "dom_set_text_i32",
])

"""
    scrub_host_globals!(mod) → Int

Returns the number of globals/declarations neutered. Idempotent —
re-running on the same module does nothing once everything is clean.
"""
function scrub_host_globals!(mod::LLVM.Module)
    n = 0
    for gv in collect(LLVM.globals(mod))
        if LLVM.isdeclaration(gv)
            nm = LLVM.name(gv)
            if startswith(nm, "jl_")
                ty = LLVM.global_value_type(gv)
                LLVM.initializer!(gv, LLVM.null(ty))
                LLVM.linkage!(gv, LLVM.API.LLVMInternalLinkage)
                n += 1
            end
            continue
        end
        init = LLVM.initializer(gv)
        init === nothing && continue
        if init isa LLVM.ConstantExpr &&
           LLVM.opcode(init) == LLVM.API.LLVMIntToPtr
            ity = LLVM.value_type(init)
            LLVM.initializer!(gv, LLVM.null(ity))
            n += 1
        end
    end

    # Julia codegen sometimes declares `@jl_int64_type`, `@jl_float32_type`
    # etc. as **functions** (`declare ptr @jl_float32_type()`) and then
    # uses them as *data* pointers (`load i64, ptr @jl_float32_type`).
    # That's an ABI mismatch: wasm-ld sees a data relocation against a
    # function symbol and aborts with `invalid data relocation:
    # jl_float32_type`. Replace each such function declaration with an
    # internal `i64` zero global so the existing loads resolve cleanly.
    for fn in collect(LLVM.functions(mod))
        LLVM.isdeclaration(fn) || continue
        nm = LLVM.name(fn)
        startswith(nm, "jl_") && endswith(nm, "_type") || continue
        # Build a same-named (after rename) i64 zero global, replace
        # uses, erase the function declaration.
        ty = LLVM.Int64Type()
        LLVM.name!(fn, "")  # free the name
        gv = LLVM.GlobalVariable(mod, ty, nm)
        LLVM.initializer!(gv, LLVM.null(ty))
        LLVM.linkage!(gv, LLVM.API.LLVMInternalLinkage)
        LLVM.replace_uses!(fn, gv)
        LLVM.erase!(fn)
        n += 1
    end

    # `@julia.get_pgcstack` returns Julia's per-thread GC frame stack
    # pointer. Wasm has no host-side runtime to provide one — and Whisk
    # uses its own GC, not Julia's. Define a trivial body that returns
    # null so calls resolve and the optimizer can usually fold them
    # away (the result is only used by GC-frame intrinsics that
    # GPUCompiler's lowering should have already stripped).
    if haskey(LLVM.functions(mod), "julia.get_pgcstack")
        fn = LLVM.functions(mod)["julia.get_pgcstack"]
        if LLVM.isdeclaration(fn)
            entry = LLVM.BasicBlock(fn, "entry")
            @dispose b = LLVM.IRBuilder() begin
                LLVM.position!(b, entry)
                LLVM.ret!(b, LLVM.null(LLVM.return_type(LLVM.function_type(fn))))
            end
            LLVM.linkage!(fn, LLVM.API.LLVMInternalLinkage)
            n += 1
        end
    end

    n += lower_julia_gc_intrinsics!(mod)
    # Rebind any `jlplt_*` thunks for which we ship a wasm-side
    # runtime body BEFORE `lower_dlsym_calls!` traps every indirect
    # `jlplt_*` call site.
    n += redirect_julia_plt_to_wasm_runtime!(mod)
    # Same idea for the inline-cached pattern (`@ccall_<sym>_NNN`):
    # Julia's frontend sometimes inlines the cached-fp + dlsym lookup
    # into every caller instead of routing through a `jlplt_*` thunk.
    # The thunk redirect doesn't help there — we need to walk the
    # cached globals directly. Must run before `lower_dlsym_calls!`
    # otherwise the indirect calls have already been replaced with
    # `js_console + trap` and there's nothing left to redirect.
    n += redirect_julia_ccall_cache_to_wasm_runtime!(mod)
    n += lower_dlsym_calls!(mod)
    return n
end

"""
    lower_dlsym_calls!(mod) → Int

Neutralise every `@ijl_load_and_lookup(libname, sym, handle) → ptr`
call. Julia emits these as `ccall(:foo, ...)` lowering when it can't
resolve the symbol at compile time — they look up `foo` in libjulia
at runtime, then *indirectly call* the result with a variadic
signature. Wasm has no varargs ABI, so LLVM-wasm SIGSEGVs in
`getAlignment` during instruction selection of the indirect call.

We replace each lookup with a load of `null`, RAUW the result, and
let the subsequent indirect call become `call %null(args)`. Then we
walk those indirect calls and replace each with `js_console(error)`
+ `unreachable` so the IR stays well-formed and the wasm validator
accepts it. At runtime, reaching one of these traps with a clear
error message — these symbols are typically in dead/error paths
that real workloads never reach.
"""
function lower_dlsym_calls!(mod::LLVM.Module)
    haskey(LLVM.functions(mod), "ijl_load_and_lookup") || return 0
    lookup_fn = LLVM.functions(mod)["ijl_load_and_lookup"]
    js_fn = _get_or_declare_js_console!(mod)
    trap_fn = _get_or_declare_trap!(mod)

    n = 0
    for use in collect(LLVM.uses(lookup_fn))
        call = LLVM.user(use)
        call isa LLVM.CallInst || continue
        # Tag the call result as null. Subsequent indirect calls through
        # it then have a null function pointer; we rewrite those next.
        rt   = LLVM.value_type(call)
        null = LLVM.null(rt)
        LLVM.replace_uses!(call, null)
        LLVM.erase!(call)
        n += 1
    end

    # Walk for indirect calls that came from `ccall(:jl_*, ...)` lowering.
    # Two characterising shapes:
    #   (a) variadic call type — `call ... (T, ...) %fp(args)`. Julia
    #       emits these for `ccall` to symbols whose Julia-side
    #       declaration is `(args::Tuple)`. Wasm has no varargs ABI;
    #       LLVM-wasm `getAlignment` segfaults on the call type.
    #   (b) callee loaded from a `@jlplt_*_got` global. Julia caches
    #       runtime-resolved symbols there. The subsequent indirect
    #       call is non-variadic but has `undef` args in some paths
    #       (the cache-miss branch traps before reaching them) which
    #       also crashes ISel.
    # Genuine function-pointer indirect calls (e.g. user-written
    # callbacks) shouldn't match either pattern.
    for fn in collect(LLVM.functions(mod))
        LLVM.isdeclaration(fn) && continue
        for bb in collect(LLVM.blocks(fn))
            for inst in collect(LLVM.instructions(bb))
                inst isa LLVM.CallInst || continue
                callee = LLVM.operands(inst)[end]
                callee isa LLVM.Function && continue
                callee isa LLVM.InlineAsm && continue
                _is_problematic_indirect_call(inst, callee) || continue
                _replace_dlsym_call_with_trap!(inst, js_fn, trap_fn, mod)
                n += 1
            end
        end
    end
    return n
end

"""
    redirect_julia_plt_to_wasm_runtime!(mod) → Int

Julia's frontend emits a Procedure Linkage Table for every
`ccall(:sym, …)` it can't statically resolve. The shape is:

    @ccall_<sym>_N        = global ptr null              ; cached fp
    @jlplt_<sym>_N_got    = global ptr @jlplt_<sym>_N #0  ; got slot, attr "julia.fname"="<sym>"
    @_j_str_<sym>_K       = constant "<sym>\\00"

    define private void @jlplt_<sym>_N(args...) \\{
      %cached = load atomic ptr, ptr @ccall_<sym>_N
      br on null → dlsym
      dlsym: %fp = call ijl_load_and_lookup(_, &name_str, _)
             store %fp → @ccall_<sym>_N
      ; merge:
      %fp = phi(cached, found)
      store %fp → @jlplt_<sym>_N_got
      call %fp(args...)
      ret void
    \\}

    ; call sites:
    %fp = load atomic ptr, ptr @jlplt_<sym>_N_got
    call %fp(args...)

The whole apparatus exists so that, on a host-VM Julia run, the
first call resolves `<sym>` via `dlopen`/`dlsym` at runtime and
caches the result. On wasm we have neither dlsym nor a host runtime
to look it up in — `lower_dlsym_calls!` would otherwise rewrite all
the indirect calls to traps.

For symbols where Whisk's own runtime *does* provide a wasm-side
body (currently `jl_genericmemory_copyto`, but the mechanism is
generic), we want the indirect calls to actually reach that body.
The cleanest place to intercept is the thunk itself: replace its
entire body with a single `tail call @<sym>(args)` and let LLVM
DCE the rest. The `_got` indirection through cached function
pointers stays in place but every load now resolves to a function
pointer that just trampolines into our wasm-side definition;
inlining usually devirtualises the call site too.

We discover thunks by walking globals named `jlplt_*_got` (whose
initializer is the thunk function itself) and parsing the symbol
out of the global's name — `jlplt_<sym>_<N>_got`. The `julia.fname`
attribute IS attached to the got global, but LLVM.jl doesn't expose
global-variable attribute access yet, so name-parsing is what we
have. Equivalent in practice: the naming is part of Julia's
codegen ABI for ccall-PLT.
"""
function redirect_julia_plt_to_wasm_runtime!(mod::LLVM.Module)
    n = 0
    for gv in collect(LLVM.globals(mod))
        gv_name = LLVM.name(gv)
        startswith(gv_name, "jlplt_") && endswith(gv_name, "_got") || continue
        target_name = parse_jlplt_got_symbol(gv_name)
        target_name === nothing && continue
        init = LLVM.initializer(gv)
        init isa LLVM.Function || continue
        thunk = init

        target = _wasm_host_import_or_runtime_target!(mod, target_name,
                                                      LLVM.function_type(thunk))
        target === nothing && continue
        target === thunk && continue
        LLVM.function_type(thunk) == LLVM.function_type(target) || continue

        rewrite_thunk_to_tail_call!(thunk, target)
        # Also rewrite the call sites loading from this GOT global so
        # `lower_dlsym_calls!`'s `@jlplt_*_got` heuristic doesn't see
        # them and replace them with traps.
        rewrite_jlplt_callsites_to_direct!(gv, target)
        n += 1
    end
    return n
end

function _wasm_host_import_or_runtime_target!(mod::LLVM.Module,
                                               name::AbstractString,
                                               fty::LLVM.FunctionType)
    if haskey(LLVM.functions(mod), name)
        target = LLVM.functions(mod)[name]
        LLVM.function_type(target) == fty || return nothing
        LLVM.isdeclaration(target) || return target
        name in _whisk_wasm_host_imports && return target
        return nothing
    end
    name in _whisk_wasm_host_imports || return nothing
    return LLVM.Function(mod, String(name), fty)
end

"""
    redirect_julia_ccall_cache_to_wasm_runtime!(mod) → Int

Sister of `redirect_julia_plt_to_wasm_runtime!` for the *inline-cached*
ccall lowering. When Julia's frontend inlines the dlsym lookup directly
into a caller (instead of going through a `jlplt_<sym>_<N>` thunk), the
shape is:

    @ccall_<sym>_<N> = global ptr null              ; cached fp

    %cached = load atomic ptr, ptr @ccall_<sym>_<N>
    %miss   = icmp eq ptr %cached, null
    br %miss, dlsym, ccall
    dlsym: %fp = call ijl_load_and_lookup(...)
           store %fp → @ccall_<sym>_<N>
           br ccall
    ccall: %fp = phi(%cached, %fp)
           %r  = call ... %fp(args)

For symbols where Whisk's own runtime *does* provide a wasm-side body,
walk every load of `@ccall_<sym>_<N>`, find the indirect-call user, and
rewrite it to a direct call to that body. The `dlsym` block, the load,
the phi and the cached global all become dead and are DCE'd.

We discover the symbol by parsing the global's name (`ccall_<sym>_<N>`)
exactly as `redirect_julia_plt_to_wasm_runtime!` does for `jlplt_*_got`.
"""
function redirect_julia_ccall_cache_to_wasm_runtime!(mod::LLVM.Module)
    n = 0
    for gv in collect(LLVM.globals(mod))
        gv_name = LLVM.name(gv)
        startswith(gv_name, "ccall_") || continue
        target_name = parse_ccall_cache_symbol(gv_name)
        target_name === nothing && continue
        haskey(LLVM.functions(mod), target_name) || continue
        target = LLVM.functions(mod)[target_name]
        LLVM.isdeclaration(target) && continue
        # Reuse the same callsite-rewriting helper as the PLT path.
        n_rw = rewrite_jlplt_callsites_to_direct!(gv, target)
        n += n_rw > 0 ? 1 : 0
    end
    return n
end

# Parse `ccall_<sym>_<N>` → `<sym>` where `<N>` is a run of digits.
# Same shape as `parse_jlplt_got_symbol` but no `_got` suffix.
function parse_ccall_cache_symbol(gv_name::AbstractString)
    startswith(gv_name, "ccall_") || return nothing
    core = gv_name[length("ccall_")+1 : end]
    last_us = findlast('_', core)
    last_us === nothing && return nothing
    digits = core[last_us+1:end]
    isempty(digits) && return nothing
    all(isdigit, digits) || return nothing
    return core[1:last_us-1]
end

# For every `%fp = load atomic ptr, ptr @<got_global>` followed by
# `call %fp(args)`, replace the call with a direct call to `target` and
# erase the load. Operates only on uses of `got_global`; non-load uses
# (e.g. the thunk's own atomic store) are left alone.
#
# Also handles the inline-cached ccall lowering, where the loaded
# function pointer feeds a PHI (one incoming = cached load, the other =
# the dlsym path which would store + reload), and the PHI is what the
# indirect call uses. We treat any CallInst that ultimately resolves
# (through PHIs) to this load as a redirection target — wasm-side
# bodies make the dlsym path dead anyway.
function rewrite_jlplt_callsites_to_direct!(got_global::LLVM.GlobalVariable,
                                              target::LLVM.Function)
    n = 0
    target_ft = LLVM.function_type(target)
    rewrite_call! = function (call::LLVM.CallInst, callee_value::LLVM.Value)
        ops = LLVM.operands(call)
        ops[end] === callee_value || return false
        args = LLVM.Value[ops[i] for i in 1:length(ops)-1]
        length(args) == length(LLVM.parameters(target)) || return false
        @dispose b=LLVM.IRBuilder() begin
            LLVM.position!(b, call)
            direct = LLVM.call!(b, target_ft, target, args)
            rt = LLVM.value_type(call)
            if rt != LLVM.VoidType()
                LLVM.replace_uses!(call, direct)
            end
        end
        LLVM.erase!(call)
        return true
    end
    for use in collect(LLVM.uses(got_global))
        load = LLVM.user(use)
        load isa LLVM.LoadInst || continue
        # Direct call sites first.
        for cu in collect(LLVM.uses(load))
            call = LLVM.user(cu)
            call isa LLVM.CallInst || continue
            rewrite_call!(call, load) && (n += 1)
        end
        # Indirect (load → phi → call) — common in the inline-cached
        # ccall lowering. Walk phi users of the load and rewrite calls
        # that use the phi as callee.
        for cu in collect(LLVM.uses(load))
            phi = LLVM.user(cu)
            phi isa LLVM.PHIInst || continue
            for pu in collect(LLVM.uses(phi))
                call = LLVM.user(pu)
                call isa LLVM.CallInst || continue
                rewrite_call!(call, phi) && (n += 1)
            end
            # Phi is dead once its calls are gone; let DCE clean it up.
        end
        # If the load now has no remaining non-store users, erase it.
        # Atomic stores to the GOT survive as dead and get DCE'd along
        # with the global itself.
        all(u -> LLVM.user(u) isa LLVM.StoreInst || LLVM.user(u) isa LLVM.PHIInst,
            collect(LLVM.uses(load))) || continue
        # Don't erase: phi/store users may still hold references; let
        # downstream DCE handle it. The important thing is the call has
        # been redirected.
    end
    return n
end

# Parse `jlplt_<sym>_<N>_got` → `<sym>` where `<N>` is a run of digits.
# The symbol itself can contain underscores (e.g. `jl_genericmemory_copyto`).
function parse_jlplt_got_symbol(gv_name::AbstractString)
    startswith(gv_name, "jlplt_") || return nothing
    endswith(gv_name, "_got") || return nothing
    # Strip prefix and suffix.
    core = gv_name[length("jlplt_")+1 : end-length("_got")]
    # Find the trailing `_<digits>` and strip.
    last_us = findlast('_', core)
    last_us === nothing && return nothing
    digits = core[last_us+1:end]
    isempty(digits) && return nothing
    all(isdigit, digits) || return nothing
    return core[1:last_us-1]
end

function rewrite_thunk_to_tail_call!(thunk::LLVM.Function, target::LLVM.Function)
    # Erase every existing block so we can rebuild a one-block body
    # that just forwards to the target. (Walking a CFG and removing
    # arbitrary blocks risks dangling phi operands; full reset is
    # the simplest correct approach for these tiny thunks.)
    for bb in collect(LLVM.blocks(thunk))
        LLVM.erase!(bb)
    end
    entry = LLVM.BasicBlock(thunk, "entry")
    @dispose b=LLVM.IRBuilder() begin
        LLVM.position!(b, entry)
        args = LLVM.Value[a for a in LLVM.parameters(thunk)]
        ret_ty = LLVM.return_type(LLVM.function_type(target))
        if ret_ty == LLVM.VoidType()
            call = LLVM.call!(b, LLVM.function_type(target), target, args)
            LLVM.tailcall!(call, true)
            LLVM.ret!(b)
        else
            call = LLVM.call!(b, LLVM.function_type(target), target, args)
            LLVM.tailcall!(call, true)
            LLVM.ret!(b, call)
        end
    end
    return thunk
end

# True if this indirect call has a shape the wasm backend can't lower:
# variadic, or callee transitively from a `@jlplt_*_got` global, or
# callee is null (a previous pass replaced the dlsym lookup).
function _is_problematic_indirect_call(inst::LLVM.CallInst, callee)
    ftype = LLVM.called_type(inst)
    ftype isa LLVM.FunctionType && LLVM.isvararg(ftype) && return true
    callee isa LLVM.Constant && LLVM.isnull(callee) && return true
    return _origin_is_jlplt(callee, Set{LLVM.Value}())
end

# DFS through phi/select/load/bitcast back to the originating value(s).
# Returns true if any path leads to a `@jlplt_*_got` global load.
function _origin_is_jlplt(v::LLVM.Value, seen::Set{LLVM.Value})
    v in seen && return false
    push!(seen, v)
    if v isa LLVM.LoadInst
        ptr_op = LLVM.operands(v)[1]
        if ptr_op isa LLVM.GlobalVariable && occursin("jlplt", LLVM.name(ptr_op))
            return true
        end
        return false
    elseif v isa LLVM.PHIInst || v isa LLVM.SelectInst
        for op in LLVM.operands(v)
            op isa LLVM.Value || continue
            _origin_is_jlplt(op, seen) && return true
        end
        return false
    end
    return false
end

function _replace_dlsym_call_with_trap!(call::LLVM.CallInst,
                                         js_fn::LLVM.Function,
                                         trap_fn::LLVM.Function,
                                         mod::LLVM.Module)
    location = _dbg_location_string(call)
    msg = (isempty(location) ? "?:?" : location) *
          ": ccall to host-VM symbol reached at runtime — not supported in wasm"
    str_gv  = _intern_message_global!(mod, msg)
    str_len = sizeof(msg)

    @dispose b = LLVM.IRBuilder() begin
        LLVM.position!(b, call)
        LLVM.call!(b, LLVM.function_type(js_fn), js_fn, LLVM.Value[
            LLVM.ConstantInt(LLVM.Int32Type(), 2),
            str_gv,
            LLVM.ConstantInt(LLVM.Int64Type(), str_len),
        ])
        LLVM.call!(b, LLVM.function_type(trap_fn), trap_fn, LLVM.Value[])
    end

    rt = LLVM.value_type(call)
    if rt != LLVM.VoidType()
        LLVM.replace_uses!(call, LLVM.UndefValue(rt))
    end
    LLVM.erase!(call)
    return
end

"""
    lower_julia_gc_intrinsics!(mod) → Int

Lower the GC-related `@julia.*` intrinsics that GPUCompiler's
toplevel optimizer normally handles, but which leak through into
sub-modules linked via `link_compiler_rt!` (toplevel = false skips
`lower_gc_frame_intrinsics!` etc.).

  * `@julia.pointer_from_objref(x)` → `x` (Julia ptrs ARE wasm ptrs
    after our addrspace strip; the conversion is a no-op).
  * `@julia.gc_loaded(mem, data)`   → `data` (the second arg is the
    already-loaded value; the intrinsic exists only to keep the
    GC-frame pass happy on the host).
  * `@julia.write_barrier(parent, kid, …)` → erased (we don't have a
    generational GC).
  * `@julia.gc_alloc_obj(task, sz, ty)` → `whisk_gc_alloc(sz)` cast
    back to ptr (mirrors `lower_gc_alloc_wasm!` in `runtime/gc.jl`,
    but reaches sub-modules linked after `finish_module!`).
"""
function lower_julia_gc_intrinsics!(mod::LLVM.Module)
    n = 0
    n += _replace_intrinsic_with_arg!(mod, "julia.pointer_from_objref", 1)
    n += _replace_intrinsic_with_arg!(mod, "julia.gc_loaded", 2)
    n += _erase_intrinsic_calls!(mod, "julia.write_barrier")
    n += _lower_gc_alloc_obj!(mod)
    return n
end

# Replace each call to `fn_name` with its `which`-th argument and
# erase the call. Useful for "identity"-shaped intrinsics like
# `julia.pointer_from_objref` and `julia.gc_loaded`.
function _replace_intrinsic_with_arg!(mod::LLVM.Module, fn_name::String, which::Int)
    haskey(LLVM.functions(mod), fn_name) || return 0
    fn = LLVM.functions(mod)[fn_name]
    n = 0
    for use in collect(LLVM.uses(fn))
        call = LLVM.user(use)
        call isa LLVM.CallInst || continue
        args = LLVM.arguments(call)
        which <= length(args) || continue
        LLVM.replace_uses!(call, args[which])
        LLVM.erase!(call)
        n += 1
    end
    return n
end

# Drop every call to `fn_name`. For void intrinsics whose only purpose
# is GC bookkeeping the host doesn't need at runtime.
function _erase_intrinsic_calls!(mod::LLVM.Module, fn_name::String)
    haskey(LLVM.functions(mod), fn_name) || return 0
    fn = LLVM.functions(mod)[fn_name]
    n = 0
    for use in collect(LLVM.uses(fn))
        call = LLVM.user(use)
        call isa LLVM.CallInst || continue
        LLVM.erase!(call)
        n += 1
    end
    return n
end

# Rewrite `@julia.gc_alloc_obj(task, sz, ty)` into a call to our own
# `whisk_gc_alloc(sz)` followed by an `inttoptr` back to ptr. The third
# argument (the type pointer) is unused — Whisk's GC is type-erased.
function _lower_gc_alloc_obj!(mod::LLVM.Module)
    fn_name = "julia.gc_alloc_obj"
    haskey(LLVM.functions(mod), fn_name) || return 0
    target = LLVM.functions(mod)[fn_name]

    # `whisk_gc_alloc(i64) -> i64` is what `runtime/gc.jl::_emit_gc_alloc!`
    # creates for top-level. If the symbol's already in the module
    # (top-level case) just reuse it; otherwise declare so the linker
    # finds it after our subsequent passes wire things up.
    alloc_fty = LLVM.FunctionType(LLVM.Int64Type(), [LLVM.Int64Type()])
    alloc_fn  = if haskey(LLVM.functions(mod), "whisk_gc_alloc")
        LLVM.functions(mod)["whisk_gc_alloc"]
    else
        LLVM.Function(mod, "whisk_gc_alloc", alloc_fty)
    end

    n = 0
    for use in collect(LLVM.uses(target))
        call = LLVM.user(use)
        call isa LLVM.CallInst || continue
        args = LLVM.arguments(call)
        length(args) >= 2 || continue
        sz = args[2]
        @dispose b = LLVM.IRBuilder() begin
            LLVM.position!(b, call)
            addr = LLVM.call!(b, alloc_fty, alloc_fn, LLVM.Value[sz])
            ptr  = LLVM.inttoptr!(b, addr, LLVM.value_type(call))
            LLVM.replace_uses!(call, ptr)
            LLVM.erase!(call)
        end
        n += 1
    end
    return n
end
