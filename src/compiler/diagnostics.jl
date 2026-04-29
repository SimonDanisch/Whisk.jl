## Trap-site diagnostics
##
## GPUCompiler's `lower_throw!` rewrites every `jl_throw(ex)` to:
##
##   call void @gpu_report_exception(ptr @exception_name)
##   call void @gpu_signal_exception()
##   call void @llvm.trap()
##   unreachable
##
## …with the message text baked in as a constant string global. That's
## fine when there's a host runtime to receive `gpu_report_exception`,
## but for our wasm output we want errors that say *where* in the user's
## Julia code they came from. JS sees a `WebAssembly.RuntimeError` either
## way; the difference is whether the developer also gets a usable
## `console.error` line first.
##
## This pass walks every `gpu_report_exception` call, reads the
## `!dbg DILocation` attached to the call, and rewrites the call to:
##
##   call void @js_console(i32 2, ptr @msg, i64 len)
##
## where `@msg` is a fresh constant `"<file>:<line>: <original message>"`.
## `gpu_signal_exception` calls become unused (we keep `llvm.trap` /
## `unreachable` as the actual halt mechanism) and `wasm-ld` drops them
## along with the `gpu_report_exception` import.

"""
    lower_dynamic_dispatch!(mod) → Int

Replace every `jl_apply_generic` / `ijl_apply_generic` call (Julia's
runtime dynamic-dispatch entry point) with a
`js_console(2, "<file>:<line>: dynamic dispatch reached …")` followed
by `llvm.trap()`. Drops the original call, so the wasm module never
needs to import it.

Compiled-with-concrete-types code never reaches these calls; if one
survives type inference, the user's program tried to do something
genuinely dynamic (heterogeneous container, `Function`-typed field,
etc.) which Whisk doesn't support yet. Trapping with a useful
`console.error` is far better than a silent `LinkError`.
"""
function lower_dynamic_dispatch!(mod::LLVM.Module)
    n = 0
    js_fn   = nothing
    trap_fn = nothing
    for fname in ("jl_apply_generic", "ijl_apply_generic")
        haskey(LLVM.functions(mod), fname) || continue
        target_fn = LLVM.functions(mod)[fname]
        for use in collect(LLVM.uses(target_fn))
            call = LLVM.user(use)
            call isa LLVM.CallInst || continue
            js_fn   === nothing && (js_fn   = _get_or_declare_js_console!(mod))
            trap_fn === nothing && (trap_fn = _get_or_declare_trap!(mod))
            _replace_dispatch_with_diagnostic!(call, js_fn, trap_fn, mod)
            n += 1
        end
    end
    return n
end

function _get_or_declare_trap!(mod::LLVM.Module)
    haskey(LLVM.functions(mod), "llvm.trap") &&
        return LLVM.functions(mod)["llvm.trap"]
    return LLVM.Function(mod, "llvm.trap",
                         LLVM.FunctionType(LLVM.VoidType()))
end

function _replace_dispatch_with_diagnostic!(call::LLVM.CallInst,
                                             js_fn::LLVM.Function,
                                             trap_fn::LLVM.Function,
                                             mod::LLVM.Module)
    location = _dbg_location_string(call)
    msg = (isempty(location) ? "?:?" : location) *
          ": dynamic dispatch (jl_apply_generic) reached at runtime — recompile with concrete types"
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

    # The original call returned `ptr` (jl_value_t*). Anything that used
    # the result is unreachable post-trap, but the IR must still type-
    # check, so replace uses with `undef` of the same type before we
    # erase the call.
    rt = LLVM.value_type(call)
    if rt != LLVM.VoidType()
        LLVM.replace_uses!(call, LLVM.UndefValue(rt))
    end
    LLVM.erase!(call)
    return
end

"""
    lower_trap_reports!(mod) → Int

Replace every `gpu_report_exception(ptr @msg)` call with a
`js_console(2, ptr @new_msg, i64 len)` call whose message includes
`<file>:<line>:` from the call's debug info. Returns the number of
calls rewritten.
"""
function lower_trap_reports!(mod::LLVM.Module)
    n = 0
    if haskey(LLVM.functions(mod), "gpu_report_exception")
        report_fn = LLVM.functions(mod)["gpu_report_exception"]
        js_fn     = _get_or_declare_js_console!(mod)
        # We may need multiple iterations: when a `gpu_report_exception`
        # call's instruction operand is itself a ConstantExpr that gets
        # rewritten, LLVM may repoint the call to a fresh constant and
        # the use list shifts mid-walk. Loop to fixpoint.
        while true
            uses_now = collect(LLVM.uses(report_fn))
            isempty(uses_now) && break
            progressed = false
            for use in uses_now
                call = LLVM.user(use)
                call isa LLVM.CallInst || continue
                _rewrite_report_to_console!(call, js_fn, mod)
                n += 1
                progressed = true
            end
            progressed || break
        end
    end
    if haskey(LLVM.functions(mod), "gpu_signal_exception")
        sig_fn = LLVM.functions(mod)["gpu_signal_exception"]
        for use in collect(LLVM.uses(sig_fn))
            call = LLVM.user(use)
            call isa LLVM.CallInst || continue
            LLVM.erase!(call)
        end
    end
    return n
end

# `void js_console(i32 level, ptr msg, i64 len)` — the wasm-import
# console channel. We materialise the declaration on demand because
# user code that never calls `js_console` directly otherwise wouldn't
# have it in the module.
function _get_or_declare_js_console!(mod::LLVM.Module)
    haskey(LLVM.functions(mod), "js_console") &&
        return LLVM.functions(mod)["js_console"]
    fty = LLVM.FunctionType(LLVM.VoidType(),
                            [LLVM.Int32Type(),
                             LLVM.PointerType(LLVM.Int8Type()),
                             LLVM.Int64Type()])
    return LLVM.Function(mod, "js_console", fty)
end

function _rewrite_report_to_console!(call::LLVM.CallInst, js_fn::LLVM.Function, mod::LLVM.Module)
    args = collect(LLVM.arguments(call))
    orig_msg::String = "exception"
    if !isempty(args)
        extracted = _extract_string_constant(args[1])
        if extracted !== nothing
            orig_msg = String(extracted)
        end
    end
    location::String = _dbg_location_string(call)
    full_msg::String = isempty(location) ? orig_msg : (location * ": " * orig_msg)

    str_gv  = _intern_message_global!(mod, full_msg)
    str_len = sizeof(full_msg)

    @dispose b = LLVM.IRBuilder() begin
        LLVM.position!(b, call)
        LLVM.call!(b, LLVM.function_type(js_fn), js_fn, LLVM.Value[
            LLVM.ConstantInt(LLVM.Int32Type(), 2),  # level = error
            str_gv,
            LLVM.ConstantInt(LLVM.Int64Type(), str_len),
        ])
        LLVM.erase!(call)
    end
    return
end

# Read the bytes of a constant string operand. Walks through pointer
# bitcasts / GEPs that LLVM may wrap a `[N x i8]` global with. Returns
# `nothing` if the operand isn't a recognisable constant string.
function _extract_string_constant(op)
    g = _peel_to_global(op)
    g === nothing && return nothing
    init = LLVM.initializer(g)
    init === nothing && return nothing
    init isa LLVM.ConstantArray || init isa LLVM.ConstantDataArray || return nothing
    raw = UInt8[convert(UInt8, c) for c in collect(init)]
    # Trim trailing NUL bytes — `globalstring_ptr!` adds one terminator.
    while !isempty(raw) && raw[end] == 0x00
        pop!(raw)
    end
    return String(raw)
end

function _peel_to_global(op)
    op isa LLVM.GlobalVariable && return op
    if op isa LLVM.ConstantExpr
        opcode = LLVM.opcode(op)
        if opcode == LLVM.API.LLVMGetElementPtr || opcode == LLVM.API.LLVMBitCast
            return _peel_to_global(LLVM.operands(op)[1])
        end
    end
    return nothing
end

function _dbg_location_string(inst::LLVM.Instruction)
    md = LLVM.metadata(inst)
    haskey(md, LLVM.MD_dbg) || return ""
    loc = md[LLVM.MD_dbg]
    # Walk the inline chain to its outermost call site — the location
    # closest to the user's source rather than to a deep stdlib helper.
    while true
        outer = LLVM.inlined_at(loc)
        outer === nothing && break
        loc = outer
    end
    sc = LLVM.scope(loc)
    sc === nothing && return ""
    f = LLVM.file(sc)
    f === nothing && return ""
    path = LLVM.filename(f)
    return "$(path):$(LLVM.line(loc))"
end

# Cache message globals in the module by name suffix so repeated traps
# at the same site share a single `[N x i8]` constant.
function _intern_message_global!(mod::LLVM.Module, msg::AbstractString)
    name = "_whisk_diag_" * string(hash(msg); base = 16)
    if haskey(LLVM.globals(mod), name)
        return LLVM.globals(mod)[name]
    end
    bytes = codeunits(msg)
    ty    = LLVM.ArrayType(LLVM.Int8Type(), length(bytes))
    init  = LLVM.ConstantDataArray(LLVM.Int8Type(), Vector{UInt8}(bytes))
    gv    = LLVM.GlobalVariable(mod, ty, name)
    LLVM.initializer!(gv, init)
    LLVM.linkage!(gv, LLVM.API.LLVMInternalLinkage)
    LLVM.constant!(gv, true)
    return gv
end
