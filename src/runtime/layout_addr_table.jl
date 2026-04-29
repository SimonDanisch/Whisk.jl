## Wasm-side: expose per-tag LayoutDescriptor addresses to JS.
##
## When the host builds an image with `convert2wasm`, every GC header
## inside the image needs a layout_ptr pointing at a wasm-rodata
## LayoutDescriptor for the corresponding type. JS resolves those
## addresses at upload time by calling
##
##     __whisk_layout_addr(tag::u64) → u64
##
## which is a switch-on-tag → return ptrtoint(@_whisk_layout_<tag>).
## The set of tags is defined by the `LayoutTagTable` passed into
## `compile_to_wasm`; each (elsz, ptr_offsets) tuple is materialized
## via `get_or_emit_layout_descriptor!` (so we share globals with any
## descriptors already emitted by gc_alloc lowering for types the
## compiled function allocates internally).

"""
    emit_layout_addr_table!(mod, tags::LayoutTagTable)

Emit an export `__whisk_layout_addr(tag::u64) → u64` that returns the
wasm-side address of the LayoutDescriptor global for `tag`. Tags
correspond to indices 0..n-1 in `tags`. A tag outside that range
hits an `unreachable` (caller bug; JS resolves tags from the manifest
emitted alongside the wasm and shouldn't request unknown ones).

Re-uses descriptor globals that gc_alloc lowering may have already
emitted for the same (elsz, offsets) keys — the per-module cache in
`runtime/layout.jl` makes this transparent.
"""
function emit_layout_addr_table!(mod::LLVM.Module, tags::LayoutTagTable)
    n = length(tags.keys)
    n == 0 && return
    T_w  = LLVM.Int64Type()

    # Materialize each tag's descriptor as a wasm-rodata global.
    desc_globals = LLVM.GlobalVariable[
        get_or_emit_layout_descriptor!(mod, elsz, copy(offsets))
        for (elsz, offsets) in tags.keys
    ]

    fn_ty = LLVM.FunctionType(T_w, [T_w])
    fn = if haskey(LLVM.functions(mod), "__whisk_layout_addr")
        LLVM.functions(mod)["__whisk_layout_addr"]
    else
        LLVM.Function(mod, "__whisk_layout_addr", fn_ty)
    end
    LLVM.linkage!(fn, LLVM.API.LLVMExternalLinkage)
    push!(LLVM.function_attributes(fn),
          LLVM.StringAttribute("wasm-export-name", "__whisk_layout_addr"))
    push!(LLVM.function_attributes(fn), LLVM.EnumAttribute("noinline", 0))

    tag_arg = LLVM.parameters(fn)[1]
    entry   = LLVM.BasicBlock(fn, "entry")
    bad_tag = LLVM.BasicBlock(fn, "bad_tag")
    case_blocks = LLVM.BasicBlock[]
    for i in 0:(n - 1)
        push!(case_blocks, LLVM.BasicBlock(fn, "case_$i"))
    end

    @dispose b=LLVM.IRBuilder() begin
        # Switch dispatcher
        LLVM.position!(b, entry)
        sw = LLVM.switch!(b, tag_arg, bad_tag, n)
        for (i, blk) in enumerate(case_blocks)
            LLVM.API.LLVMAddCase(sw, LLVM.ConstantInt(T_w, i - 1), blk)
        end

        # Each case: ret ptrtoint(@_whisk_layout_<tag> to i64)
        for (i, blk) in enumerate(case_blocks)
            LLVM.position!(b, blk)
            addr = LLVM.ptrtoint!(b, desc_globals[i], T_w)
            LLVM.ret!(b, addr)
        end

        # Bad tag → trap. Caller bug; deliberately loud.
        LLVM.position!(b, bad_tag)
        trap_fn = _get_or_declare_trap!(mod)
        LLVM.call!(b, LLVM.function_type(trap_fn), trap_fn, LLVM.Value[])
        LLVM.unreachable!(b)
    end

    LLVM.API.LLVMAppendToUsed(mod, [fn], 1)
    return fn
end
