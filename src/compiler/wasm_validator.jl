## Pre-emit IR validator
##
## Walks the LLVM module right before `LLVMTargetMachineEmitToMemoryBuffer`
## and reports IR shapes the wasm backend can't lower. Without this,
## those shapes manifest as `signal 11 (1): Segmentation fault` deep
## inside `llvm::DataLayout::getAlignment` — entirely unhelpful for
## diagnosing user code.
##
## The validator is structurally similar to GPUCompiler's `check_ir`
## but tuned to the failure modes we've actually hit:
##   * variadic indirect calls (`call ... (T, ...) %fp(...)`)
##   * indirect calls whose callee transitively comes from a
##     `@jlplt_*_got` global (PLT-GOT)
##   * leftover `@julia.*` intrinsic calls (gc_alloc_obj,
##     pointer_from_objref, gc_loaded, write_barrier, call2)
##   * unhandled `@ijl_apply_generic` / `@ijl_invoke` calls
##   * external function declarations referenced from code but not
##     in our import whitelist
##
## For each finding it tries to extract a `file:line` via the call's
## `!dbg DILocation` and emits a single-line diagnostic. Set the env
## var `WHISK_VALIDATE_STRICT=1` to make findings throw; otherwise
## they print as warnings (and `LLVMTargetMachineEmit` may still
## crash — but at least the user knows what to look at).

"""
    validate_for_wasm!(mod) → Vector{String}

Run the wasm pre-emit validator on `mod`. Returns a list of
diagnostic strings (empty when clean). Throws if the env var
`WHISK_VALIDATE_STRICT` is set to a truthy value AND any diagnostics
were produced.
"""
function validate_for_wasm!(mod::LLVM.Module)
    diags = String[]
    seen  = Dict{String, Int}()  # de-dupe by reason → count

    _push! = (reason::String) -> begin
        seen[reason] = get(seen, reason, 0) + 1
    end

    for fn in LLVM.functions(mod)
        LLVM.isdeclaration(fn) && continue
        for bb in LLVM.blocks(fn)
            for inst in LLVM.instructions(bb)
                inst isa LLVM.CallInst || continue
                _classify_call!(inst, fn, _push!)
            end
        end
    end

    # Whitelist of imports we *expect* to be left undeclared. Everything
    # else with `external` linkage and no body is suspect.
    expected_imports = Set([
        "js_console", "memcpy", "memmove", "memset",
        "gpu_report_exception", "gpu_signal_exception",
        "whisk_dynamic_dispatch_unreached",
    ])
    for fn in LLVM.functions(mod)
        LLVM.isdeclaration(fn) || continue
        nm = LLVM.name(fn)
        startswith(nm, "llvm.") && continue
        nm in expected_imports && continue
        # Functions only declared but never called shouldn't reach
        # the wasm backend — wasm-ld DCEs them. So we only complain
        # about declarations that *do* have call sites. `LLVM.uses`
        # returns an iterator of `LLVM.Use` (non-bool), so we can't
        # just `any(...)` it — check emptiness instead.
        iterate(LLVM.uses(fn)) === nothing && continue
        _push!("undeclared external `@$(nm)` is referenced — wasm-ld will reject this as an undefined symbol")
    end

    # Materialise diagnostics. Sorted for deterministic output.
    for (reason, count) in sort(collect(seen))
        push!(diags, count == 1 ? reason : "$(reason)  ×$(count)")
    end

    if !isempty(diags)
        msg = "Whisk: pre-emit validation found $(length(diags)) issue(s) the wasm backend probably can't lower:\n" *
              join("  • " .* diags, "\n")
        if _strict_mode()
            error(msg)
        else
            @warn msg
        end
    end
    return diags
end

function _strict_mode()
    v = get(ENV, "WHISK_VALIDATE_STRICT", "")
    !isempty(v) && lowercase(v) ∉ ("0", "false", "no", "off")
end

# Classify a single call instruction. Pushes one or more diagnostic
# strings into the accumulator.
function _classify_call!(call::LLVM.CallInst, parent_fn::LLVM.Function,
                          push_diag::Function)
    callee = LLVM.operands(call)[end]

    # Direct call to a known function — usually fine. Only flag if the
    # callee is one of the unhandled `@julia.*` intrinsics or known
    # bad ijl_* runtime entry points that should have been lowered.
    if callee isa LLVM.Function
        nm = LLVM.name(callee)
        if startswith(nm, "julia.") &&
                !(nm in ("julia.get_pgcstack",))   # we provide a stub for this
            push_diag(_with_loc(call,
                "leftover `@$(nm)` intrinsic — should have been lowered " *
                "(check `lower_julia_call2!` / `lower_julia_gc_intrinsics!`)"))
        elseif nm in ("ijl_invoke", "ijl_apply_generic")
            push_diag(_with_loc(call,
                "leftover `@$(nm)` call — should have been replaced by " *
                "`lower_invokes!` (static) or `lower_dynamic_dispatch!` (dynamic)"))
        elseif nm in ("ijl_load_and_lookup", "ijl_get_nth_field_checked")
            push_diag(_with_loc(call,
                "`@$(nm)` call survives — Julia runtime symbol that wasm can't resolve. " *
                "Likely an unresolved `ccall(:sym, …)` — replace with a `Whisk` runtime " *
                "helper (link via `link_compiler_rt!` / `Base.llvmcall` direct, not `ccall`)"))
        end
        return
    end
    callee isa LLVM.InlineAsm && return  # asm blocks pass through to wasm-ld

    # Indirect call. Check the shapes wasm-LLVM can't handle.
    ftype = LLVM.called_type(call)
    if ftype isa LLVM.FunctionType && LLVM.isvararg(ftype)
        push_diag(_with_loc(call,
            "variadic indirect call (`call ... (..., ...) %fp(...)`) — wasm has no " *
            "varargs ABI; LLVM-wasm `getAlignment` SIGSEGVs on this. Replace the " *
            "underlying `ccall(:sym, …)` with a Whisk runtime helper or trap-stub."))
        return
    end

    if callee isa LLVM.Constant && LLVM.isnull(callee)
        push_diag(_with_loc(call,
            "indirect call through a null function pointer — call site reachable but " *
            "callee was nulled out (probably by `lower_dlsym_calls!`). Adjust the " *
            "trap-rewrite to also replace the call site."))
        return
    end

    if _origin_is_jlplt(callee, Set{LLVM.Value}())
        push_diag(_with_loc(call,
            "indirect call through `@jlplt_*_got` cache — Julia's runtime symbol-resolution " *
            "pattern. The wasm backend can't lower these. `lower_dlsym_calls!` should be " *
            "trapping them; check filter logic."))
        return
    end

    # Other indirect calls (real function-pointer args) — usually OK,
    # but flag for attention.
    push_diag(_with_loc(call,
        "indirect call to non-Function callee (type = $(ftype)) — verify wasm-ld can lower this"))
end

function _with_loc(inst::LLVM.Instruction, msg::String)
    loc = _dbg_location_string(inst)
    fnname = LLVM.name(LLVM.parent(LLVM.parent(inst)))
    isempty(loc) ?
        "in `$(fnname)`: $(msg)" :
        "$(loc) [in `$(fnname)`]: $(msg)"
end
