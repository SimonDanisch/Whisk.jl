## Compile + link pipeline
##
## Modeled on Abacus's compiler/compilation.jl, with the final link step
## swapped from a host shared library to a wasm module.

"""
    compile_to_wasm(f, argtypes; out_path, name=nothing, optimize=true,
                    additional_exports=())

Compile a Julia function `f(args::argtypes...)` into a `.wasm` module at
`out_path`. Returns a NamedTuple `(; wasm_path, entry_name, llvm_ir)`.

`additional_exports` is an iterable of `(; f, name)` NamedTuples
specifying additional Julia helper functions that should also be
exported by the wasm module. Each helper must be reachable from the
main entry's call graph (typically by being called from a `@noinline`
dispatcher). After GPUCompiler compiles the main entry, we walk the
LLVM module looking for the helper's Julia-mangled name (matching
prefix `julia_<funcname>_`) and add a `wasm-export-name` attribute +
pin via `@llvm.used` so the helper appears as its own wasm export.

This approach avoids sub-compile (which trips Julia's `@julia.call2`
lowering inconsistency between toplevel and sub jobs). Trade-off: the
helpers MUST be reachable from the main entry, so a dispatcher is
still required even though JS can call the helpers directly.
"""
function compile_to_wasm(@nospecialize(f), @nospecialize(argtypes::Type{<:Tuple});
                         out_path::String,
                         name::Union{Nothing,String} = nothing,
                         optimize::Bool = true,
                         additional_exports = (),
                         layout_tags = nothing)   # ::Union{Nothing, LayoutTagTable} — type defined later in include order
    # Closures (callable structs with captured fields) compile fine, but the
    # resulting wasm function takes the closure object as its first arg.
    # Auto-marshal: encode `f` into an image alongside the wasm, generate a
    # `@__whisk_closure_root` global + setter, wrap the user-facing entry to
    # inject the closure pointer. Singleton functions skip this entirely.
    has_closure = !Base.issingletontype(typeof(f))
    closure_image = UInt8[]
    closure_tags  = nothing
    if has_closure
        closure_image, closure_tags = encode_closure_image(f)
        # Merge closure-emitted tags into the user-supplied table so the GC
        # marker has descriptors for every field type reachable via the
        # closure root. layout_tags is `LayoutTagTable | Nothing` — we can't
        # name the type yet (it's defined later in include order), so let
        # merge_layout_tags figure out the union at call time.
        layout_tags = merge_layout_tags(layout_tags, closure_tags)
    end

    # Phase 1 — Julia source → LLVM IR via GPUCompiler.
    target = WasmCompilerTarget()
    params = WasmCompilerParams()
    config = CompilerConfig(target, params;
                            kernel = false,
                            name   = name,
                            always_inline = false,
                            # GPUCompiler's check_ir hardcodes rejection of
                            # `jl_invoke`, `jl_apply_generic`, and `tojlinvoke*`
                            # trampolines — that's a GPU-runtime constraint, not
                            # a wasm one. Disable it; we run our own
                            # `validate_ir(::WasmCompilerJob, …)` for wasm-
                            # specific checks. Dead boxing inside throw
                            # constructors gets DCE'd by LLVM after
                            # `lower_throw!`, so the runtime calls usually
                            # don't survive.
                            validate = false,
                            # Whisk owns its runtime: `finish_module!` lowers
                            # every `julia.gc_alloc_obj` to `whisk_gc_alloc`
                            # before GPUCompiler's driver would link its GPU
                            # runtime library, and `link_compiler_rt!` provides
                            # `ijl_box_*` as plain Julia helpers backed by the
                            # Whisk GC. Linking GPUCompiler's rtlib anyway
                            # would bind throw-path box calls to ITS
                            # `ijl_box_*` (renamed `.N` on collision), which
                            # allocate via a host-imported `gpu_malloc` and
                            # tag with host-VM type pointers — wrong allocator
                            # AND wrong tags for our GC. So: don't link it.
                            libraries = false,
                            opt_level = optimize ? 2 : 0)
    src    = methodinstance(typeof(f), argtypes)
    job    = CompilerJob(src, config)

    # Closed-world reachability (ARCHITECTURE.md phase 1): the method/
    # type universe drives dynamic-dispatch resolution. Host-side
    # analysis only — failures degrade to the trap lowering.
    world = try
        reachable_world(f, argtypes)
    catch err
        @warn "Whisk: reachability analysis failed; dynamic dispatch falls back to traps" err
        nothing
    end

    JuliaContext() do _ctx
        mod, meta = GPUCompiler.compile(:llvm, job)
        # Sub-compile memo entries are keyed by module objectid; a
        # previous compile that threw mid-pipeline never reached its
        # end-of-pipeline cleanup. Sweep here — compiles don't nest or
        # overlap — so failed runs can't pin MethodInstances forever.
        empty!(SUB_COMPILE_MEMO)
        entry_fn = _resolve_entry_name(mod, meta, name)

        # Mark the entry point as exported so wasm-ld keeps it visible.
        entry_f = LLVM.functions(mod)[entry_fn]
        push!(LLVM.function_attributes(entry_f),
              LLVM.StringAttribute("wasm-export-name", entry_fn))

        # Closure case: the original entry takes (closure_ptr, args...).
        # Replace the user-facing export with a thin wrapper that loads
        # the closure pointer from `@__whisk_closure_root` and forwards.
        # The original entry stays in the module under a private name.
        if has_closure
            emit_closure_wrapper!(mod, entry_f, entry_fn)
        end

        # Tag any additional helper functions reachable from the main
        # entry's call graph with their requested wasm-export-name.
        # This must run AFTER GPUCompiler.compile but BEFORE optimize
        # passes inline/DCE the helpers — the @llvm.used pin keeps
        # them alive thereafter.
        for entry_spec in additional_exports
            attach_additional_export!(mod, entry_spec.f, String(entry_spec.name))
        end

        # Link in compiler-rt helpers (`__multi3`, `__lshrti3`, `__ashlti3`,
        # `ijl_box_*`) so LLVM's late lowering / Julia-IR finds them instead
        # of leaving unresolved imports. Each helper is a plain Julia
        # function compiled via a sub-job, then renamed to its C ABI symbol.
        link_compiler_rt!(mod, target, params)

        # The rt-helpers may bring in their own `gpu_report_exception`
        # call sites (any throw path inside e.g. `whisk_jl_box_*`),
        # `jl_apply_generic` references, host-pointer literals in global
        # initializers, external `@jl_*` declarations, and unresolved
        # `@julia.call2(@ijl_invoke, …)` wrappers (their toplevel=false
        # sub-job skipped Julia's lowering). Re-run the finalizer passes
        # here so the linked module is wasm-clean. Order matters:
        # `lower_julia_call2!` must run before `lower_invokes!` (it
        # produces the @ijl_invoke shape `lower_invokes!` consumes), and
        # both must run before `scrub_host_globals!` neuters the
        # `@jl_global_*` MI pointers they read.
        # Each lower_invokes! run sub-compiles and links the targets
        # of @ijl_invoke calls — but those targets can have their own
        # @ijl_invoke calls (deep dispatch chains, e.g. Marshal.decode
        # over a struct of Vectors of structs). Loop to fixpoint so
        # nested invokes get fully resolved before we move on.
        # The fixpoint includes closed-world dispatch resolution:
        # every stage's sub-compiles can link modules carrying fresh
        # `julia.call` wrappers, `ijl_invoke`s AND `jl_apply_generic`
        # sites — iterate until nothing new appears.
        converged = false
        fpdbg = haskey(ENV, "WHISK_DEBUG_DISPATCH")
        for it in 1:64
            t0 = time()
            lower_julia_call2!(mod)
            nb = lower_julia_call_builtins!(mod, world, target, params)
            ni = lower_invokes!(mod, target, params)
            na = world === nothing ? 0 :
                 lower_apply_generic!(mod, world, target, params)
            fpdbg && @info "fixpoint[$it]: builtins=$nb invokes=$ni apply_generic=$na " *
                "fns=$(count(Returns(true), LLVM.functions(mod))) dt=$(round(time() - t0; digits=1))s"
            if (nb + ni + na) == 0
                converged = true
                break
            end
        end
        converged ||
            @warn "Whisk: lowering fixpoint hit the iteration cap; unresolved calls will trap"
        # Define the `whisk_typetag_<T>` externs the linked `ijl_box_*`
        # helpers reference. After the invoke fixpoint (so every
        # declaration is in), before host-pointer rewriting (which
        # creates the same `_whisk_type_*` descriptor globals for
        # baked DataType pointers — both paths must converge on one
        # global per type).
        define_box_typetags!(mod)
        # Lower any leftover sub-compiled alloc sites (linked rt helpers,
        # invoked-method bodies) BEFORE host-pointer constants get
        # zeroed — that's the only window where we can decode the
        # type-pointer literals into precise GC layout descriptors.
        lower_gc_alloc_wasm!(mod)
        lower_alloc_genericmemory!(mod)
        rewrite_host_pointer_constants!(mod)
        # Subtype checks against abstract types decide per concrete
        # universe member; needs descriptor-global RHS operands, hence
        # after the host-pointer rewrite.
        world === nothing || lower_subtype_checks!(mod, world)
        scrub_host_globals!(mod)
        lower_dynamic_dispatch!(mod)
        # Any ijl_invoke whose MI never decoded: loud trap, not import.
        lower_undecoded_invokes!(mod)
        lower_trap_reports!(mod)
        # Per-tag descriptor address lookup, if the caller provided a
        # LayoutTagTable. Must run before clear_layout_cache! so the
        # descriptor cache is still warm (it shares globals with any
        # already emitted by gc_alloc lowering for types this function
        # internally allocates).
        if layout_tags !== nothing
            emit_layout_addr_table!(mod, layout_tags)
        end
        Whisk.clear_layout_cache!(mod)
        delete!(SUB_COMPILE_MEMO, objectid(mod))

        LLVM.verify(mod)

        # If the env var WHISK_DUMP_IR is set, write the about-to-be-emitted
        # LLVM IR to that path. Useful for diagnosing wasm-backend crashes
        # (the wasm-ld step or `LLVMTargetMachineEmitToMemoryBuffer`
        # SIGSEGV that hits us when an internal type lookup fails). The
        # dump goes here, NOT after `emit`, so when emit segfaults we
        # still have the IR.
        let dump = get(ENV, "WHISK_DUMP_IR", "")
            isempty(dump) || (write(dump, string(mod)); @info "Whisk: dumped IR" path=dump)
        end

        # Pre-emit IR validator: catches the IR shapes we know LLVM-wasm
        # can't lower (variadic indirect calls, `@jlplt_*_got` callees,
        # leftover `@julia.*` intrinsics, undeclared `@ijl_*` runtime
        # symbols) and reports them with `file:line` instead of letting
        # `LLVMTargetMachineEmitToMemoryBuffer` SIGSEGV in
        # `getAlignment`. Set `WHISK_VALIDATE_STRICT=1` to make any
        # finding throw rather than warn.
        validate_for_wasm!(mod)

        # Phase 2 — LLVM IR → wasm object file via the LLVM wasm32 backend.
        tm  = GPUCompiler.llvm_machine(target)
        obj = LLVM.emit(tm, mod, LLVM.API.LLVMObjectFile)

        ir_text = string(mod)

        # Phase 3 — wasm object → linked wasm module via lld.
        obj_path = out_path * ".o"
        write(obj_path, obj)
        link_extras = String[]
        if layout_tags !== nothing && !isempty(layout_tags.keys)
            push!(link_extras, "__whisk_layout_addr")
        end
        for entry_spec in additional_exports
            push!(link_extras, String(entry_spec.name))
        end
        if has_closure
            push!(link_extras, "whisk_set_closure_root")
        end
        try
            link_wasm(obj_path, out_path;
                      entry_name = entry_fn,
                      extra_exports = link_extras)
        finally
            rm(obj_path; force = true)
        end

        manifest = manifest_from_layout_tags(layout_tags, entry_fn)

        return (; wasm_path = out_path, entry_name = entry_fn, llvm_ir = ir_text,
                  closure_image = closure_image, manifest = manifest)
    end
end

"""
    manifest_from_layout_tags(tags, entry_name) → NamedTuple | Nothing

Build the JS-side manifest (`tags::Vector{NamedTuple}`, `entry_name`) from a
`LayoutTagTable`. Returns `nothing` if `tags === nothing`. The manifest's
`tags` array is indexed by tag id (i - 1 for the i-th type) and lets the
JS-side `WhiskInstance._resolveLayouts()` look up each layout descriptor's
runtime address via the wasm-exported `__whisk_layout_addr(tag)` helper.
"""
function manifest_from_layout_tags(tags, entry_name::String)
    tags === nothing && return nothing
    return (
        tags = [(tag = i - 1,
                 name = string(tags.types[i]),
                 elsz = Int(tags.keys[i][1]),
                 ptr_offsets = Int.(tags.keys[i][2]))
                for i in 1:length(tags.keys)],
        entry_name = entry_name,
    )
end

"""
    encode_closure_image(f) → (bytes::Vector{UInt8}, tags::LayoutTagTable)

Encode a callable struct `f` (a closure with captured fields) into a
Whisk image blob. The image's root is the closure object's objref —
i.e. the pointer wasm code expects when receiving the closure as the
implicit first argument.

Reuses the standard `image_builder` machinery: each captured field
recurses through `encode!` / `_emit_inline!`, producing addr fixups
for heap-ref fields (Vectors, Strings, mutable structs, …) and inline
bytes for bitstype fields. The returned `tags` table lists every type
whose layout descriptor must be addressable in the wasm module so the
GC marker can walk objects reached via the closure.
"""
function encode_closure_image(f)
    w = ImageWriter()
    root = encode!(w, f)
    w.root_offset = root
    return _serialize_image(w), w.tags
end

"""
    merge_layout_tags(a, b) → LayoutTagTable | Nothing

Merge two `LayoutTagTable`s (or `nothing`) into one. `nothing` acts as
identity. Order: `a`'s entries come first; `b`'s new tags are appended
with their original tag IDs preserved unless they collide.

Used by `compile_to_wasm` when both the caller and the closure encoder
contribute layout tags.
"""
function merge_layout_tags(a, b)
    a === nothing && return b
    b === nothing && return a
    out = a
    for (T, tag) in pairs(b.keys)
        haskey(out.keys, T) && continue
        out.keys[T] = tag
    end
    return out
end

"""
    emit_closure_wrapper!(mod, inner_fn, export_name)

Replace the user-facing entry function with a thin wrapper that loads
the closure object pointer from a wasm-resident global and forwards to
the original Julia-compiled body. The original body retains its IR
name (renamed to `<export_name>__inner`) and its `wasm-export-name`
attribute is stripped — only the wrapper is exported.

Also emits:

  * `@__whisk_closure_root` — a global `i64` holding the closure objref
    pointer. Initialised to 0; the JS host writes the post-upload
    address before the first call.
  * `@whisk_set_closure_root(i64)` — exported setter, called by JS once
    after `uploadImage(closure_image)` returns.

Calling convention: Julia's GPUCompiler passes the closure object as
the first parameter (typed `ptr` after `lower_addrspace_casts`), then
the user-facing args. The wrapper's signature is
`(<user args>) → <ret>`, the inner is `(ptr, <user args>) → <ret>`.
"""
function emit_closure_wrapper!(mod::LLVM.Module, inner_fn::LLVM.Function,
                                 export_name::String)
    inner_ft = LLVM.function_type(inner_fn)
    inner_params = LLVM.parameters(inner_fn)
    length(inner_params) >= 1 ||
        error("emit_closure_wrapper!: closure entry expected ≥1 parameter (the closure object), got 0")
    user_param_tys = LLVM.LLVMType[LLVM.value_type(p) for p in inner_params[2:end]]
    ret_ty = LLVM.return_type(inner_ft)

    # Strip the wasm-export-name attribute on the inner so only the
    # wrapper is exposed. Rename the inner so the wrapper can take its
    # original name (= export_name).
    attrs = LLVM.function_attributes(inner_fn)
    for a in collect(attrs)
        if a isa LLVM.StringAttribute && LLVM.kind(a) == "wasm-export-name"
            LLVM.delete!(attrs, a)
        end
    end
    LLVM.name!(inner_fn, export_name * "__inner")

    # The closure-root global. i64 (wasm64) — same width as a wasm ptr.
    root_gv = LLVM.GlobalVariable(mod, LLVM.Int64Type(), "__whisk_closure_root")
    LLVM.initializer!(root_gv, LLVM.ConstantInt(LLVM.Int64Type(), 0))
    LLVM.linkage!(root_gv, LLVM.API.LLVMInternalLinkage)

    # Setter: stored value is the wasm address of the closure objref.
    setter_ft = LLVM.FunctionType(LLVM.VoidType(), [LLVM.Int64Type()])
    setter_fn = LLVM.Function(mod, "whisk_set_closure_root", setter_ft)
    push!(LLVM.function_attributes(setter_fn),
          LLVM.StringAttribute("wasm-export-name", "whisk_set_closure_root"))
    push!(LLVM.function_attributes(setter_fn), LLVM.EnumAttribute("noinline", 0))
    LLVM.linkage!(setter_fn, LLVM.API.LLVMExternalLinkage)
    LLVM.API.LLVMAppendToUsed(mod, [setter_fn], 1)
    let
        bb = LLVM.BasicBlock(setter_fn, "entry")
        @dispose b=LLVM.IRBuilder() begin
            LLVM.position!(b, bb)
            LLVM.store!(b, LLVM.parameters(setter_fn)[1], root_gv)
            LLVM.ret!(b)
        end
    end

    # Wrapper: loads closure pointer, calls inner with (ptr, user_args...).
    wrap_ft = LLVM.FunctionType(ret_ty, user_param_tys)
    wrap_fn = LLVM.Function(mod, export_name, wrap_ft)
    push!(LLVM.function_attributes(wrap_fn),
          LLVM.StringAttribute("wasm-export-name", export_name))
    push!(LLVM.function_attributes(wrap_fn), LLVM.EnumAttribute("noinline", 0))
    LLVM.linkage!(wrap_fn, LLVM.API.LLVMExternalLinkage)
    LLVM.API.LLVMAppendToUsed(mod, [wrap_fn], 1)

    bb = LLVM.BasicBlock(wrap_fn, "entry")
    @dispose b=LLVM.IRBuilder() begin
        LLVM.position!(b, bb)
        root_i64 = LLVM.load!(b, LLVM.Int64Type(), root_gv)
        # The inner's first parameter is `ptr` (after addrspace cleanup).
        # Build it from the i64 via inttoptr.
        inner_ptr_ty = LLVM.value_type(inner_params[1])
        root_ptr = LLVM.inttoptr!(b, root_i64, inner_ptr_ty)
        call_args = LLVM.Value[root_ptr]
        for p in LLVM.parameters(wrap_fn)
            push!(call_args, p)
        end
        call = LLVM.call!(b, inner_ft, inner_fn, call_args)
        LLVM.tailcall!(call, true)
        if ret_ty == LLVM.VoidType()
            LLVM.ret!(b)
        else
            LLVM.ret!(b, call)
        end
    end
    return wrap_fn
end

"""
    link_wasm(obj_path, wasm_path; entry_name)

Turn a wasm object file into a final wasm module via `wasm-ld`. We
vendor LLD 21.1.8 as a Whisk artifact (see `Artifacts.toml`) rather
than rely on `LLD_jll`: Julia 1.12 stdlib pins `LLD_jll` to v18.1.7,
which has a wasm64 bug that mis-reads relocations for any function
emitting an sret/byval struct argument and aborts with
`invalid relocation data index`. LLD 19+ fixes that.
"""
function link_wasm(obj_path::String, wasm_path::String; entry_name::String,
                   initial_memory::Integer = 16 * 1024 * 1024,                 # 16 MB
                   max_memory::Integer     = Int(4) * 1024 * 1024 * 1024,      # 4 GB (wasm64)
                   extra_exports::Vector{String} = String[])
    lld     = _whisk_wasm_ld_path()
    lib_dir = _whisk_libllvm_lib_dir()
    base_args = String[
        "-mwasm64",
        "--no-entry",
        # Keep all data (notably type descriptors) above 4096: Julia's
        # tag regime treats values < 4096 as small typetags, so a
        # descriptor address below that would be misread by any
        # `tag < 4096` branch. (lld's default global base is 1024.)
        "--global-base=65536",
        "--export=$entry_name",
        "--export=whisk_gc_alloc",
        "--export=whisk_gc_reserve",
        "--export=whisk_gc_collect",
        "--export=whisk_pin",
        "--export=whisk_unpin",
    ]
    for e in extra_exports
        push!(base_args, "--export=$e")
    end
    push!(base_args, "--allow-undefined",
                     "--initial-memory=$initial_memory",
                     "--max-memory=$max_memory",
                     "-o", wasm_path,
                     obj_path)
    cmd = Cmd(vcat([lld], base_args))
    cmd = addenv(cmd, "LD_LIBRARY_PATH" => _ld_library_path_with(lib_dir))
    run(cmd)
    return wasm_path
end

const _WASM_LD_PATH = Ref{String}()
const _LIBLLVM_LIB_DIR = Ref{String}()

function _whisk_wasm_ld_path()
    isassigned(_WASM_LD_PATH) && return _WASM_LD_PATH[]
    Pkg.Artifacts.ensure_artifact_installed("LLD", _whisk_artifacts_toml())
    art = Pkg.Artifacts.artifact_path(
        Pkg.Artifacts.artifact_hash("LLD", _whisk_artifacts_toml()))
    candidates = (joinpath(art, "tools", "wasm-ld"),
                  joinpath(art, "bin",   "wasm-ld"))
    for c in candidates
        if isfile(c)
            _WASM_LD_PATH[] = c
            return c
        end
    end
    error("Whisk: vendored LLD artifact has no wasm-ld at $(candidates)")
end

function _whisk_libllvm_lib_dir()
    isassigned(_LIBLLVM_LIB_DIR) && return _LIBLLVM_LIB_DIR[]
    Pkg.Artifacts.ensure_artifact_installed("libLLVM", _whisk_artifacts_toml())
    art = Pkg.Artifacts.artifact_path(
        Pkg.Artifacts.artifact_hash("libLLVM", _whisk_artifacts_toml()))
    lib = joinpath(art, "lib")
    isdir(lib) || error("Whisk: vendored libLLVM artifact has no lib/ at $(art)")
    _LIBLLVM_LIB_DIR[] = lib
    return lib
end

_whisk_artifacts_toml() = normpath(joinpath(@__DIR__, "..", "..", "Artifacts.toml"))

function _ld_library_path_with(extra::AbstractString)
    cur = get(ENV, "LD_LIBRARY_PATH", "")
    isempty(cur) ? extra : "$(extra):$(cur)"
end

"""
    attach_additional_export!(mod, f, export_name) → LLVM.Function or nothing

Find the LLVM function compiled for Julia function `f` in `mod` (by
matching its Julia-mangled name, typically `julia_<funcname>_<id>`)
and tag it with `wasm-export-name=export_name`. Pin via `@llvm.used`
so the optimizer doesn't DCE it. Returns the tagged function, or
`nothing` if no matching function was found (in which case the
helper wasn't reachable from the main entry's call graph and the
caller should ensure a `@noinline` dispatcher references it).
"""
function attach_additional_export!(mod::LLVM.Module, @nospecialize(f),
                                     export_name::String)
    funcname = string(nameof(f))
    prefix = "julia_$(funcname)_"
    candidates = LLVM.Function[]
    for fn in LLVM.functions(mod)
        LLVM.isdeclaration(fn) && continue
        nm = LLVM.name(fn)
        startswith(nm, prefix) && push!(candidates, fn)
    end
    if isempty(candidates)
        @warn "Whisk: additional_exports could not find a Julia-compiled function for `$funcname`. Make sure it's @noinline and called from the main entry's call graph." export_name
        return nothing
    end
    if length(candidates) > 1
        @warn "Whisk: multiple matches for additional_exports `$funcname` — using the first. Mark the helper @noinline + reference once from the dispatcher to disambiguate." matches=[LLVM.name(c) for c in candidates] export_name
    end
    fn = candidates[1]
    LLVM.name!(fn, export_name)
    LLVM.linkage!(fn, LLVM.API.LLVMExternalLinkage)
    push!(LLVM.function_attributes(fn),
          LLVM.StringAttribute("wasm-export-name", export_name))
    push!(LLVM.function_attributes(fn), LLVM.EnumAttribute("noinline", 0))
    LLVM.API.LLVMAppendToUsed(mod, [fn], 1)
    return fn
end

"""
    link_compiler_rt!(mod, target, params)

For every entry in `COMPILER_RT_HELPERS`, sub-compile the Julia helper to
LLVM IR and link the resulting function into `mod` under the C ABI symbol
name. The libcalls (`__multi3` etc.) aren't visible in the IR at this
point — they're emitted by LLVM's late instruction selection — so we link
the helpers unconditionally with external linkage. wasm-ld's dead-strip
drops unused ones from the final module.
"""
function link_compiler_rt!(mod::LLVM.Module, target, params)
    for (sym, fn, sig) in COMPILER_RT_HELPERS
        # toplevel = false → finish_module! / finish_ir! skip our hooks.
        config = CompilerConfig(target, params;
                                kernel = false,
                                name   = nothing,
                                toplevel = false,
                                always_inline = true,
                                opt_level = 2)
        src    = methodinstance(typeof(fn), sig)
        sub_job = CompilerJob(src, config)
        sub_mod, sub_meta = GPUCompiler.compile(:llvm, sub_job)

        # Rename the Julia entry to the C ABI symbol so LLVM's i128
        # lowering can resolve to our definition instead of leaving a
        # libcall import.
        sub_entry = sub_meta.entry
        LLVM.name!(sub_entry, sym)
        LLVM.linkage!(sub_entry, LLVM.API.LLVMExternalLinkage)

        # Lower Julia call wrappers BEFORE the strip pass: sub-jobs
        # (toplevel=false) skip finish_ir!, so `@julia.call2` wrappers
        # and the corresponding `@ijl_invoke` calls would otherwise
        # ride along into the linked module. They reference
        # `@jl_global_NNN` ConstantExpr initializers that the strip
        # pipeline (`RemoveJuliaAddrspacesPass`) tends to neuter — once
        # neutered we can't decode the host-VM MI pointer back, and
        # `lower_invokes!` fails for those calls. So: lower here, while
        # the inits are still `inttoptr (i64 ADDR to ptr)`.
        lower_julia_call2!(sub_mod)
        lower_invokes!(sub_mod, target, params)

        # Same strip-and-cleanup pipeline that `_sub_compile_mi!` runs
        # on lowered-`ijl_invoke` sub-modules — see invoke_lowering.jl
        # for the rationale.
        _strip_julia_addrspaces!(sub_mod, target)

        LLVM.link!(mod, sub_mod)
    end
    return
end

"""
    _resolve_entry_name(mod, meta, requested_name)

Find the entry function's current name in `mod`, robust to passes that
may have recreated the function and invalidated `meta.entry`. Falls back
in order: requested user name → `meta.entry.name` if still valid → the
sole external-linkage non-runtime define in the module.
"""
function _resolve_entry_name(mod::LLVM.Module, meta, requested_name)
    if requested_name isa AbstractString && haskey(LLVM.functions(mod), String(requested_name))
        return String(requested_name)
    end
    nm = LLVM.name(meta.entry)
    if !isempty(nm) && haskey(LLVM.functions(mod), nm)
        return nm
    end
    # Last resort: scan for an externally-linked define that isn't one of
    # the runtime helpers we link in. There should be exactly one when
    # `requested_name === nothing`.
    for fn in LLVM.functions(mod)
        LLVM.isdeclaration(fn) && continue
        LLVM.linkage(fn) == LLVM.API.LLVMExternalLinkage || continue
        n = LLVM.name(fn)
        startswith(n, "whisk_") && continue
        n in ("__multi3", "__lshrti3", "__ashlti3") && continue
        return n
    end
    error("Whisk: could not locate entry function in compiled module")
end

"""
    methodinstance(ft, tt)

Resolve `Tuple{ft, tt.parameters...}` to a `MethodInstance`. Mirrors
GPUCompiler's internal helper but lives here so we don't depend on its
non-exported API.
"""
function methodinstance(@nospecialize(ft), @nospecialize(tt))
    sig = Tuple{ft, tt.parameters...}
    match = Base._which(sig)
    return Core.Compiler.specialize_method(match)
end
