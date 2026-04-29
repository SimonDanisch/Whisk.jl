## Garbage collector — precise mark-sweep with conservative scanning of
## payload bodies. wasm64 target.
##
## Linear-memory layout after wasm-ld:
##
##     0  ┌────────────────────────────────┐
##        │ rodata, data segments          │
##        ├────────────────────────────────┤
##        │ wasm stack (grows down)        │
##  __heap_base ───────────────────────────┤
##        │ allocated heap (header+payload)│
##        │   ↓ bump allocator             │
##  _whisk_heap_next ──────────────────────┤
##        │ free space                     │
##     ↓  └────────────────────────────────┘
##
## Object layout (16-byte header + payload):
##
##     ┌──────────────────────────────────┐
##     │ size       : u64                 │ — total bytes incl hdr,
##     │                                  │   multiple of 8
##     │ layout|flgs: u64                 │ — bit 0 = mark
##     │                                  │   bit 1 = free
##     │                                  │   bits 3..63 = layout_ptr
##     │                                  │     (LayoutDescriptor*; 0
##     │                                  │      means raw bytes,
##     │                                  │      mark walker skips scan)
##     ├──────────────────────────────────┤
##     │ payload …                         │
##     └──────────────────────────────────┘
##
## LayoutDescriptors live in wasm rodata (one per (elsz, ptr_offsets)
## tuple, emitted by `runtime/layout.jl`). The mark walker uses them
## to scan ONLY the byte offsets that actually hold heap pointers —
## no conservative payload scan, no false-positive resurrection.
##
## We target wasm64 (64-bit pointers and linear-memory addressing).
## This matches Julia's host-time `Int = Int64` and `sizeof(Ptr) == 8`,
## eliminating the cross-compile struct-layout inconsistencies wasm32
## suffers from. V8 and Firefox both support wasm64.
##
## Roots come from Julia's GC frame intrinsics (`julia.{new,push,pop}_
## gc_frame`, `julia.get_gc_frame_slot`, `julia.get_pgcstack`) — we
## lower them into real linked-list operations on a shadow root stack
## kept at the wasm global `_whisk_pgcstack`. Payload bodies are walked
## PRECISELY using per-type LayoutDescriptors emitted at compile time
## (see `runtime/layout.jl`). Mark uses an explicit work-stack rather
## than recursion so deep object graphs can't blow the wasm call stack.

# ── Constants ─────────────────────────────────────────────────────────

const HEADER_SIZE      = UInt64(16)
const HEADER_FLAGS_OFF = UInt64(8)
const FLAG_MARK        = UInt64(0x1)
const FLAG_FREE        = UInt64(0x2)
const MIN_PAYLOAD      = UInt64(8)   # at least 8 bytes — fits free-list link

# Mark stack capacity. Each entry is one i64 (a payload-base address).
# 256K entries × 8 bytes = 2 MB. Enough for any reasonable graph; if
# you trap with the diagnostic "mark stack overflow", bump this.
const MARK_STACK_CAP   = UInt64(64 * 1024)

# Capacity of the JS-roots table — pointers JS holds across calls and
# pins via `whisk_pin` so mark+sweep doesn't reclaim them. JNI's global
# reference table and Lua's pseudo-stack are both fixed-size for the
# same reason: pin counts are tiny in practice (graph + params), and a
# fixed cap turns misuse into a loud trap rather than silent heap
# corruption. Bumping this is mechanical if a real app ever needs more.
const JS_ROOTS_CAP     = 256

"""
    emit_wasm_runtime!(mod::LLVM.Module)

Inject the entire GC runtime into `mod`:

  * `@__heap_base`           — extern, supplied by wasm-ld.
  * `@_whisk_heap_next`      — i64 mutable global, lazy-init to
                               `&__heap_base`.
  * `@_whisk_pgcstack`       — i64 mutable global, top-of-stack ptr
                               for the GC root linked list.
  * `@_whisk_free_list_head` — i64 mutable global, head of free list.
  * `whisk_gc_alloc(sz)`              — header-aware allocator.
  * `jl_alloc_genericmemory_unchecked` — Julia 1.11+ Memory{T} ctor.
  * `whisk_gc_collect()`              — mark + sweep entry point.
  * `_whisk_gc_mark()`, `_whisk_gc_sweep()`,
    `_whisk_mark_obj_at_addr(addr)`   — internal helpers.
"""
function emit_wasm_runtime!(mod::LLVM.Module)
    T_i8 = LLVM.Int8Type()

    heap_base = if haskey(LLVM.globals(mod), "__heap_base")
        LLVM.globals(mod)["__heap_base"]
    else
        gv = LLVM.GlobalVariable(mod, T_i8, "__heap_base")
        LLVM.linkage!(gv, LLVM.API.LLVMExternalLinkage)
        LLVM.extinit!(gv, true)
        gv
    end

    heap_next = _get_or_make_i64_global!(mod, "_whisk_heap_next", 0)
    pgcstack  = _get_or_make_i64_global!(mod, "_whisk_pgcstack",  0)
    free_list = _get_or_make_i64_global!(mod, "_whisk_free_list_head", 0)
    js_roots, js_roots_len = _get_or_make_js_roots_globals!(mod)

    # GC stats counters. Updated by alloc/sweep/collect, queryable by JS
    # via the exported `whisk_gc_stat_*` getters. Useful for production
    # monitoring: rising `layout_zero_allocs` means a type pointer
    # couldn't be decoded at compile time and the object's pointer fields
    # aren't being marked — that path is a correctness risk, not a
    # performance one.
    _get_or_make_i64_global!(mod, "_whisk_gc_bytes_allocated", 0)
    _get_or_make_i64_global!(mod, "_whisk_gc_num_collections", 0)
    _get_or_make_i64_global!(mod, "_whisk_gc_live_bytes",      0)
    _get_or_make_i64_global!(mod, "_whisk_gc_layout_zero_allocs", 0)

    mark_stack, mark_top = _get_or_make_mark_stack_globals!(mod)

    _emit_gc_alloc!(mod, heap_base, heap_next, free_list)
    _emit_gc_alloc_memory!(mod)
    _emit_mark_obj_at_addr!(mod, heap_base, heap_next, mark_stack, mark_top)
    _emit_mark_drain!(mod, mark_stack, mark_top)
    _emit_gc_mark!(mod, pgcstack, js_roots, js_roots_len, mark_top)
    _emit_gc_sweep!(mod, heap_base, heap_next, free_list)
    _emit_gc_collect!(mod)
    _emit_js_pin!(mod, js_roots, js_roots_len)
    _emit_js_unpin!(mod, js_roots, js_roots_len)
    _emit_jl_genericmemory_copyto!(mod)
    _emit_jl_genericmemory_to_string!(mod)
    _emit_jl_string_to_genericmemory!(mod)
    _emit_ijl_pchar_to_string!(mod)
    _emit_ijl_alloc_string!(mod)
    _emit_jl_value_ptr!(mod)
    _emit_gc_stat_getters!(mod)
end

# Wasm-exported zero-arg getters — JS calls `instance.exports.whisk_gc_stat_*()`
# and gets back a BigInt counter. Cheap (one global load), suitable for
# polling between calls.
function _emit_gc_stat_getters!(mod::LLVM.Module)
    T_w = LLVM.Int64Type()
    for (export_name, global_name) in (
        ("whisk_gc_bytes_allocated", "_whisk_gc_bytes_allocated"),
        ("whisk_gc_num_collections", "_whisk_gc_num_collections"),
        ("whisk_gc_live_bytes",      "_whisk_gc_live_bytes"),
        ("whisk_gc_layout_zero_allocs", "_whisk_gc_layout_zero_allocs"),
    )
        haskey(LLVM.functions(mod), export_name) && continue
        fn = LLVM.Function(mod, export_name,
                            LLVM.FunctionType(T_w, LLVM.LLVMType[]))
        LLVM.linkage!(fn, LLVM.API.LLVMExternalLinkage)
        push!(LLVM.function_attributes(fn),
              LLVM.StringAttribute("wasm-export-name", export_name))
        push!(LLVM.function_attributes(fn), LLVM.EnumAttribute("noinline", 0))
        gv = LLVM.globals(mod)[global_name]
        entry = LLVM.BasicBlock(fn, "entry")
        @dispose b=LLVM.IRBuilder() begin
            LLVM.position!(b, entry)
            v = LLVM.load!(b, T_w, gv)
            LLVM.ret!(b, v)
        end
        LLVM.API.LLVMAppendToUsed(mod, [fn], 1)
    end
end

# Julia's `Base.unsafe_copyto!(::MemoryRef{T}, ::MemoryRef{T}, n)` in
# `base/genericmemory.jl:120-130` splits on `isbitstype(T)`:
#   bitstype     → `memmove(destp, srcp, sizeof(T) * n)`
#   non-bitstype → `ccall(:jl_genericmemory_copyto, Cvoid,
#                         (Any, Ptr{Cvoid}, Any, Ptr{Cvoid}, Int),
#                         dest.mem, dest.ptr_or_offset,
#                         src.mem,  src.ptr_or_offset, Int(n))`
# The non-bitstype branch fires whenever `Vector{T}` grows for a
# `T` that contains another `Vector` (e.g. `RouteSegment`, `Tile`).
# The ccall target is unlowered host runtime → traps at runtime.
#
# Element layout in `Memory{T}` for non-bitstype T:
#   * `arrayelem_isboxed=true`: each slot holds an 8-byte pointer.
#   * inline-stored struct with pointer fields (e.g. `Tile`):
#     each slot holds the struct inline, sizeof(T) bytes per slot.
# Either way, copying is just `memmove(destp, srcp, n * elsz)`.
#
# We don't have static T at this layer, so derive `elsz` at runtime
# from the dest Memory's GC header. Memory layout in linear memory:
#   gc_header (16B) ┃ length::i64 ┃ data::ptr ┃ inline payload[len*elsz]
#                   ↑ dest_mem     +0          +8         +16
# The GC header at `dest_mem - 16` stores the total block size in
# its first 8 bytes (see `_emit_gc_alloc!`). For non-bitstype T,
# `elsz` is a multiple of 8, so `len*elsz` is already 8-aligned and
# the GC allocator adds no padding — the formula
#   elsz = (block_size - 32) / length
# is exact for every Memory we emit. (Bitstype-T Memorys with
# unaligned `len*elsz` would round up, but that branch is the
# `memmove` branch in `unsafe_copyto!` and never reaches here.)
function _emit_jl_genericmemory_copyto!(mod::LLVM.Module)
    T_w   = LLVM.Int64Type()
    T_i8p = LLVM.PointerType(LLVM.Int8Type())

    fn = if haskey(LLVM.functions(mod), "jl_genericmemory_copyto")
        LLVM.functions(mod)["jl_genericmemory_copyto"]
    else
        ft = LLVM.FunctionType(LLVM.VoidType(),
                               [T_i8p, T_i8p, T_i8p, T_i8p, T_w])
        LLVM.Function(mod, "jl_genericmemory_copyto", ft)
    end
    LLVM.isdeclaration(fn) || return
    LLVM.linkage!(fn, LLVM.API.LLVMInternalLinkage)
    push!(LLVM.function_attributes(fn), LLVM.EnumAttribute("alwaysinline", 0))

    # Julia frontend lowers the ccall to:
    # `call void @jl_genericmemory_copyto(ptr %dst_mem, ptr %destp,
    #                                      ptr %src_mem, ptr %srcp, i64 %n)`
    dest_mem, destp, _src_mem, srcp, n = LLVM.parameters(fn)
    entry = LLVM.BasicBlock(fn, "entry")

    memmove_fn = if haskey(LLVM.functions(mod), "memmove")
        LLVM.functions(mod)["memmove"]
    else
        ft = LLVM.FunctionType(T_i8p, [T_i8p, T_i8p, T_w])
        f = LLVM.Function(mod, "memmove", ft)
        LLVM.linkage!(f, LLVM.API.LLVMExternalLinkage)
        f
    end

    @dispose b=LLVM.IRBuilder() begin
        LLVM.position!(b, entry)
        # length::i64 lives at dest_mem + 0
        length_v = LLVM.load!(b, T_w, dest_mem)
        # block_size::i64 at dest_mem - 16 (GC header byte 0)
        hdr_ptr = LLVM.gep!(b, LLVM.Int8Type(), dest_mem,
                             [LLVM.ConstantInt(T_w, -16)])
        block_size = LLVM.load!(b, T_w, hdr_ptr)
        # payload bytes inline = block_size - 32 (GC hdr 16 + mem hdr 16)
        payload_bytes = LLVM.sub!(b, block_size, LLVM.ConstantInt(T_w, 32))
        # elsz = payload_bytes / length (length > 0 — caller guards n==0,
        # and a 0-length Memory wouldn't be the dest of a copy)
        elsz = LLVM.udiv!(b, payload_bytes, length_v)
        nbytes = LLVM.mul!(b, n, elsz)
        LLVM.call!(b, LLVM.function_type(memmove_fn), memmove_fn,
                    [destp, srcp, nbytes])
        LLVM.ret!(b)
    end

    # Until `redirect_julia_plt_to_wasm_runtime!` rewrites the PLT thunk
    # to call this directly, the symbol has zero in-IR uses and
    # GlobalDCE during optimize! would drop it. Pin via `@llvm.used`.
    LLVM.API.LLVMAppendToUsed(mod, [fn], 1)
end

# Julia's `String(::Vector{UInt8})` and `unsafe_string(::Ptr{UInt8}, len)`
# ccall through `:jl_genericmemory_to_string` and `:ijl_pchar_to_string`
# respectively. Both have C signature `ptr fn(ptr src, i64 len)` and
# return a freshly-allocated Julia String that owns the bytes.
#
# Julia String layout (verified empirically — `pointer_from_objref(s)`
# reads as `[i64 length, u8 utf8_bytes…]`):
#
#     [GC header 16 B]
#     [length:i64 @ +0]
#     [utf8_bytes @ +8 .. +8+length]
#
# The two functions only differ in where the source bytes live:
#   * `ijl_pchar_to_string(char_ptr, len)` — `char_ptr` IS the bytes.
#   * `jl_genericmemory_to_string(mem_ptr, len)` — `mem_ptr` is a
#     Memory{UInt8} objref; the inline payload starts at `mem_ptr + 16`
#     (after the {length, data_ptr} header).
#
# Both share an "alloc + write length + memcpy + return objref" body;
# we factor it through `_emit_string_constructor!`. Layout descriptor =
# 0 (raw bytes, no pointers) — String has no heap-pointer fields.
#
# Workaround for Whisk task #20: the original symbols come in as
# unlowered PLT thunks (`jlplt_jl_genericmemory_to_string_NNN` etc) with
# call_indirect ABI Whisk-LLVM can't lower. Emitting these wasm-side
# bodies + `redirect_julia_plt_to_wasm_runtime!` makes the call land on
# real linkable functions that wasm-ld emits cleanly.

function _emit_string_constructor!(mod::LLVM.Module, fn_name::String,
                                     src_data_offset::Int)
    T_w   = LLVM.Int64Type()
    T_i8p = LLVM.PointerType(LLVM.Int8Type())

    fn = if haskey(LLVM.functions(mod), fn_name)
        LLVM.functions(mod)[fn_name]
    else
        LLVM.Function(mod, fn_name,
            LLVM.FunctionType(T_i8p, [T_i8p, T_w]))
    end
    LLVM.isdeclaration(fn) || return
    LLVM.linkage!(fn, LLVM.API.LLVMInternalLinkage)
    push!(LLVM.function_attributes(fn), LLVM.EnumAttribute("alwaysinline", 0))

    src_ptr, len = LLVM.parameters(fn)
    entry = LLVM.BasicBlock(fn, "entry")

    alloc_fn = LLVM.functions(mod)["whisk_gc_alloc"]   # (i64) -> i64
    memcpy_fn = if haskey(LLVM.functions(mod), "memcpy")
        LLVM.functions(mod)["memcpy"]
    else
        f = LLVM.Function(mod, "memcpy",
            LLVM.FunctionType(T_i8p, [T_i8p, T_i8p, T_w]))
        LLVM.linkage!(f, LLVM.API.LLVMExternalLinkage)
        f
    end

    @dispose b=LLVM.IRBuilder() begin
        LLVM.position!(b, entry)
        # Payload = 8 bytes (length field) + len bytes (UTF-8 data).
        payload = LLVM.add!(b, len, LLVM.ConstantInt(T_w, 8))
        addr_int = LLVM.call!(b, LLVM.function_type(alloc_fn), alloc_fn, [payload])
        addr_ptr = LLVM.inttoptr!(b, addr_int, T_i8p)
        # Write length::i64 at offset 0.
        LLVM.store!(b, len, addr_ptr)
        # Copy bytes from `src_ptr + src_data_offset` into `addr_ptr + 8`.
        src_data = LLVM.gep!(b, LLVM.Int8Type(), src_ptr,
                              [LLVM.ConstantInt(T_w, src_data_offset)])
        body_dst = LLVM.gep!(b, LLVM.Int8Type(), addr_ptr,
                              [LLVM.ConstantInt(T_w, 8)])
        LLVM.call!(b, LLVM.function_type(memcpy_fn), memcpy_fn,
                    [body_dst, src_data, len])
        LLVM.ret!(b, addr_ptr)
    end

    LLVM.API.LLVMAppendToUsed(mod, [fn], 1)
end

_emit_jl_genericmemory_to_string!(mod) =
    _emit_string_constructor!(mod, "jl_genericmemory_to_string", 16)
_emit_ijl_pchar_to_string!(mod) =
    _emit_string_constructor!(mod, "ijl_pchar_to_string", 0)

# `Base.StringMemory(n)` ccalls `:ijl_alloc_string` to allocate a String
# of `n` UTF-8 bytes (uninitialised). Same shape as the constructors
# above except there's no source buffer to copy from. Sig: `(i64 n) → ptr`.
function _emit_ijl_alloc_string!(mod::LLVM.Module)
    T_w   = LLVM.Int64Type()
    T_i8p = LLVM.PointerType(LLVM.Int8Type())

    fn = if haskey(LLVM.functions(mod), "ijl_alloc_string")
        LLVM.functions(mod)["ijl_alloc_string"]
    else
        LLVM.Function(mod, "ijl_alloc_string",
            LLVM.FunctionType(T_i8p, [T_w]))
    end
    LLVM.isdeclaration(fn) || return
    LLVM.linkage!(fn, LLVM.API.LLVMInternalLinkage)
    push!(LLVM.function_attributes(fn), LLVM.EnumAttribute("alwaysinline", 0))

    len = LLVM.parameters(fn)[1]
    entry = LLVM.BasicBlock(fn, "entry")
    alloc_fn = LLVM.functions(mod)["whisk_gc_alloc"]

    @dispose b=LLVM.IRBuilder() begin
        LLVM.position!(b, entry)
        payload = LLVM.add!(b, len, LLVM.ConstantInt(T_w, 8))
        addr_int = LLVM.call!(b, LLVM.function_type(alloc_fn), alloc_fn, [payload])
        addr_ptr = LLVM.inttoptr!(b, addr_int, T_i8p)
        LLVM.store!(b, len, addr_ptr)
        LLVM.ret!(b, addr_ptr)
    end

    LLVM.API.LLVMAppendToUsed(mod, [fn], 1)
end

# Inverse of `jl_genericmemory_to_string`: takes a String objref (whose
# bytes live at +8 with length at +0) and returns a fresh Memory{UInt8}
# (16 B {length, data_ptr} header + inline payload). Julia's runtime
# would alias when safe; we always copy — simpler, and the call sites
# that hit this path (e.g. inside `sprint`/`print(::Float64)`) are
# infrequent enough that the copy cost is negligible.
function _emit_jl_string_to_genericmemory!(mod::LLVM.Module)
    T_w   = LLVM.Int64Type()
    T_i8p = LLVM.PointerType(LLVM.Int8Type())

    fn = if haskey(LLVM.functions(mod), "jl_string_to_genericmemory")
        LLVM.functions(mod)["jl_string_to_genericmemory"]
    else
        LLVM.Function(mod, "jl_string_to_genericmemory",
            LLVM.FunctionType(T_i8p, [T_i8p]))
    end
    LLVM.isdeclaration(fn) || return
    LLVM.linkage!(fn, LLVM.API.LLVMInternalLinkage)
    push!(LLVM.function_attributes(fn), LLVM.EnumAttribute("alwaysinline", 0))

    str_ptr = LLVM.parameters(fn)[1]
    entry = LLVM.BasicBlock(fn, "entry")

    alloc_fn  = LLVM.functions(mod)["whisk_gc_alloc"]
    memcpy_fn = if haskey(LLVM.functions(mod), "memcpy")
        LLVM.functions(mod)["memcpy"]
    else
        f = LLVM.Function(mod, "memcpy",
            LLVM.FunctionType(T_i8p, [T_i8p, T_i8p, T_w]))
        LLVM.linkage!(f, LLVM.API.LLVMExternalLinkage)
        f
    end

    @dispose b=LLVM.IRBuilder() begin
        LLVM.position!(b, entry)
        # Read String length at offset 0.
        len = LLVM.load!(b, T_w, str_ptr)
        # Memory payload = 16 B (Memory header: length + data_ptr) + len bytes.
        payload = LLVM.add!(b, len, LLVM.ConstantInt(T_w, 16))
        addr_int = LLVM.call!(b, LLVM.function_type(alloc_fn), alloc_fn, [payload])
        mem_ptr = LLVM.inttoptr!(b, addr_int, T_i8p)
        # Memory.length at +0
        LLVM.store!(b, len, mem_ptr)
        # Memory.ptr (as i64) at +8 → mem_ptr + 16 (start of inline data).
        data_ptr  = LLVM.gep!(b, LLVM.Int8Type(), mem_ptr, [LLVM.ConstantInt(T_w, 16)])
        ptr_slot  = LLVM.gep!(b, LLVM.Int8Type(), mem_ptr, [LLVM.ConstantInt(T_w, 8)])
        data_addr = LLVM.ptrtoint!(b, data_ptr, T_w)
        LLVM.store!(b, data_addr, ptr_slot)
        # Copy bytes from String+8 into Memory's inline payload.
        src_data = LLVM.gep!(b, LLVM.Int8Type(), str_ptr, [LLVM.ConstantInt(T_w, 8)])
        LLVM.call!(b, LLVM.function_type(memcpy_fn), memcpy_fn,
                    [data_ptr, src_data, len])
        LLVM.ret!(b, mem_ptr)
    end

    LLVM.API.LLVMAppendToUsed(mod, [fn], 1)
end

# `Base.unsafe_pointer_to_objref(p::Ptr{Cvoid})::Any` ccalls
# `:jl_value_ptr` which on the host is a no-op cast (just type-tags
# the address). On wasm we have no type tags — we just need the
# address to flow through unchanged. Emit a body that returns its
# argument identity-wise; the PLT thunk for `jl_value_ptr` gets
# rewired through `redirect_julia_plt_to_wasm_runtime!`.
#
# Used by SimiMaps's `route_query` to recover a typed `RoutingGraph`
# reference from a u64 handle that JS pinned across calls — avoids
# decoding the 71 MB graph blob on every routing request.
function _emit_jl_value_ptr!(mod::LLVM.Module)
    T_w   = LLVM.Int64Type()
    T_i8p = LLVM.PointerType(LLVM.Int8Type())

    fn = if haskey(LLVM.functions(mod), "jl_value_ptr")
        LLVM.functions(mod)["jl_value_ptr"]
    else
        ft = LLVM.FunctionType(T_i8p, [T_i8p])
        LLVM.Function(mod, "jl_value_ptr", ft)
    end
    LLVM.isdeclaration(fn) || return
    LLVM.linkage!(fn, LLVM.API.LLVMInternalLinkage)
    push!(LLVM.function_attributes(fn), LLVM.EnumAttribute("alwaysinline", 0))

    arg = LLVM.parameters(fn)[1]
    entry = LLVM.BasicBlock(fn, "entry")
    @dispose b=LLVM.IRBuilder() begin
        LLVM.position!(b, entry)
        LLVM.ret!(b, arg)
    end

    # Like jl_genericmemory_copyto: pinned via @llvm.used to survive
    # GlobalDCE before the PLT redirect rewires call sites.
    LLVM.API.LLVMAppendToUsed(mod, [fn], 1)
end


# `_whisk_js_roots` — fixed-size `[JS_ROOTS_CAP x i64]` static global.
# Slots 0..len hold pointers JS pinned via `whisk_pin`. The precise
# walk in `_whisk_gc_mark` only reads slots 0..len; slots beyond len
# are not consulted. `whisk_unpin` zeroes the vacated tail slot
# defensively so that even a future conservative scan of the table
# won't keep formerly-pinned objects alive via stale bytes.
function _get_or_make_js_roots_globals!(mod::LLVM.Module)
    arr_ty = LLVM.ArrayType(LLVM.Int64Type(), JS_ROOTS_CAP)
    roots = if haskey(LLVM.globals(mod), "_whisk_js_roots")
        LLVM.globals(mod)["_whisk_js_roots"]
    else
        gv = LLVM.GlobalVariable(mod, arr_ty, "_whisk_js_roots")
        LLVM.initializer!(gv, LLVM.null(arr_ty))
        LLVM.linkage!(gv, LLVM.API.LLVMInternalLinkage)
        gv
    end
    roots_len = _get_or_make_i64_global!(mod, "_whisk_js_roots_len", 0)
    return roots, roots_len
end

function _get_or_make_i64_global!(mod, name, init::Integer)
    if haskey(LLVM.globals(mod), name)
        return LLVM.globals(mod)[name]
    end
    T = LLVM.Int64Type()
    gv = LLVM.GlobalVariable(mod, T, name)
    LLVM.initializer!(gv, LLVM.ConstantInt(T, init))
    LLVM.linkage!(gv, LLVM.API.LLVMInternalLinkage)
    return gv
end

# Mark stack — fixed-size buffer of i64 (payload-base addresses), top
# pointer counts entries (not bytes). Lives in linear memory with
# wasm rodata.
function _get_or_make_mark_stack_globals!(mod::LLVM.Module)
    arr_ty = LLVM.ArrayType(LLVM.Int64Type(), MARK_STACK_CAP)
    stack = if haskey(LLVM.globals(mod), "_whisk_mark_stack")
        LLVM.globals(mod)["_whisk_mark_stack"]
    else
        gv = LLVM.GlobalVariable(mod, arr_ty, "_whisk_mark_stack")
        LLVM.initializer!(gv, LLVM.null(arr_ty))
        LLVM.linkage!(gv, LLVM.API.LLVMInternalLinkage)
        LLVM.alignment!(gv, 8)
        gv
    end
    top = _get_or_make_i64_global!(mod, "_whisk_mark_top", 0)
    return stack, top
end

# ─────────────────────────────────────────────────────────────────────
# whisk_gc_alloc — free-list-first allocator with header.
# ─────────────────────────────────────────────────────────────────────

function _emit_gc_alloc!(mod, heap_base, heap_next, free_list)
    _emit_gc_alloc_typed!(mod, heap_base, heap_next, free_list)
    _emit_gc_alloc_untyped_wrapper!(mod)
    _emit_gc_reserve!(mod, heap_base, heap_next)
end

# `whisk_gc_alloc_typed(sz, layout_ptr) → payload_addr` is the workhorse.
# Stores `layout_ptr` (with flags=0) at header offset 8. Free-list-first
# with bump fallback. Internal linkage; called by `lower_gc_alloc_wasm!`
# rewrites and by `jl_alloc_genericmemory_unchecked`.
function _emit_gc_alloc_typed!(mod, heap_base, heap_next, free_list)
    T_w   = LLVM.Int64Type()
    T_wp  = LLVM.PointerType(T_w)

    fn = if haskey(LLVM.functions(mod), "whisk_gc_alloc_typed")
        LLVM.functions(mod)["whisk_gc_alloc_typed"]
    else
        LLVM.Function(mod, "whisk_gc_alloc_typed",
                      LLVM.FunctionType(T_w, [T_w, T_w]))
    end
    LLVM.isdeclaration(fn) || return
    LLVM.linkage!(fn, LLVM.API.LLVMInternalLinkage)
    push!(LLVM.function_attributes(fn), LLVM.EnumAttribute("noinline", 0))

    ensure_fn = if haskey(LLVM.functions(mod), "whisk_ensure_capacity")
        LLVM.functions(mod)["whisk_ensure_capacity"]
    else
        LLVM.Function(mod, "whisk_ensure_capacity",
                      LLVM.FunctionType(LLVM.VoidType(), [T_w]))
    end

    sz_param     = LLVM.parameters(fn)[1]
    layout_param = LLVM.parameters(fn)[2]
    entry  = LLVM.BasicBlock(fn, "entry")
    f_loop = LLVM.BasicBlock(fn, "freelist_loop")
    f_check= LLVM.BasicBlock(fn, "freelist_check")
    f_take = LLVM.BasicBlock(fn, "freelist_take")
    f_next = LLVM.BasicBlock(fn, "freelist_next")
    bump   = LLVM.BasicBlock(fn, "bump")
    done   = LLVM.BasicBlock(fn, "done")

    bytes_allocated_gv = LLVM.globals(LLVM.parent(fn))["_whisk_gc_bytes_allocated"]
    layout_zero_gv     = LLVM.globals(LLVM.parent(fn))["_whisk_gc_layout_zero_allocs"]

    @dispose b=LLVM.IRBuilder() begin
        LLVM.position!(b, entry)
        sz_min = _i64_max!(b, sz_param, LLVM.ConstantInt(T_w, MIN_PAYLOAD))
        sz_hdr = LLVM.add!(b, sz_min, LLVM.ConstantInt(T_w, HEADER_SIZE + 7))
        total  = LLVM.and!(b, sz_hdr,
                            LLVM.ConstantInt(T_w, reinterpret(Int64, ~UInt64(7))))

        # Stats: cumulative bytes allocated; count of layout=0 allocs
        # (raw byte buffers from JS — expected — plus any silent
        # fallback from compile-time decode failure — bug indicator).
        ba_old = LLVM.load!(b, T_w, bytes_allocated_gv)
        ba_new = LLVM.add!(b, ba_old, total)
        LLVM.store!(b, ba_new, bytes_allocated_gv)
        is_zero_layout = LLVM.icmp!(b, LLVM.API.LLVMIntEQ, layout_param,
                                     LLVM.ConstantInt(T_w, 0))
        zl_delta = LLVM.select!(b, is_zero_layout,
                                 LLVM.ConstantInt(T_w, 1),
                                 LLVM.ConstantInt(T_w, 0))
        zl_old = LLVM.load!(b, T_w, layout_zero_gv)
        zl_new = LLVM.add!(b, zl_old, zl_delta)
        LLVM.store!(b, zl_new, layout_zero_gv)

        prev_link0 = LLVM.ptrtoint!(b, free_list, T_w)
        cur0       = LLVM.load!(b, T_w, free_list)
        LLVM.br!(b, f_loop)

        LLVM.position!(b, f_loop)
        cur       = LLVM.phi!(b, T_w)
        prev_link = LLVM.phi!(b, T_w)
        is_zero = LLVM.icmp!(b, LLVM.API.LLVMIntEQ, cur,
                              LLVM.ConstantInt(T_w, 0))
        LLVM.br!(b, is_zero, bump, f_check)

        LLVM.position!(b, f_check)
        cur_size = LLVM.load!(b, T_w, LLVM.inttoptr!(b, cur, T_wp))
        fits = LLVM.icmp!(b, LLVM.API.LLVMIntUGE, cur_size, total)
        LLVM.br!(b, fits, f_take, f_next)

        LLVM.position!(b, f_take)
        cur_payload_addr = LLVM.add!(b, cur, LLVM.ConstantInt(T_w, HEADER_SIZE))
        cur_payload_ptr  = LLVM.inttoptr!(b, cur_payload_addr, T_wp)
        next_free        = LLVM.load!(b, T_w, cur_payload_ptr)
        prev_link_ptr    = LLVM.inttoptr!(b, prev_link, T_wp)
        LLVM.store!(b, next_free, prev_link_ptr)
        # Replace the free-list link/FLAG_FREE word with the
        # layout pointer (flags = 0).
        flags_addr_take  = LLVM.add!(b, cur, LLVM.ConstantInt(T_w, HEADER_FLAGS_OFF))
        flags_ptr_take   = LLVM.inttoptr!(b, flags_addr_take, T_wp)
        LLVM.store!(b, layout_param, flags_ptr_take)
        take_value       = cur_payload_addr
        LLVM.br!(b, done)

        LLVM.position!(b, f_next)
        cur_pl_addr2  = LLVM.add!(b, cur, LLVM.ConstantInt(T_w, HEADER_SIZE))
        new_prev_link = cur_pl_addr2
        new_cur       = LLVM.load!(b, T_w,
                                    LLVM.inttoptr!(b, cur_pl_addr2, T_wp))
        LLVM.br!(b, f_loop)

        LLVM.position!(b, bump)
        cur_hn    = LLVM.load!(b, T_w, heap_next)
        is_unset  = LLVM.icmp!(b, LLVM.API.LLVMIntEQ, cur_hn,
                                LLVM.ConstantInt(T_w, 0))
        base_addr = LLVM.ptrtoint!(b, heap_base, T_w)
        actual_hn = LLVM.select!(b, is_unset, base_addr, cur_hn)
        new_hn    = LLVM.add!(b, actual_hn, total)
        LLVM.call!(b, LLVM.function_type(ensure_fn), ensure_fn,
                    LLVM.Value[new_hn])
        LLVM.store!(b, new_hn, heap_next)
        LLVM.store!(b, total,
                       LLVM.inttoptr!(b, actual_hn, T_wp))
        flags_addr_bump = LLVM.add!(b, actual_hn, LLVM.ConstantInt(T_w, HEADER_FLAGS_OFF))
        LLVM.store!(b, layout_param,
                       LLVM.inttoptr!(b, flags_addr_bump, T_wp))
        bump_payload    = LLVM.add!(b, actual_hn, LLVM.ConstantInt(T_w, HEADER_SIZE))
        LLVM.br!(b, done)

        LLVM.position!(b, done)
        result = LLVM.phi!(b, T_w)
        LLVM.append!(LLVM.incoming(result),
                     [(take_value, f_take), (bump_payload, bump)])
        LLVM.ret!(b, result)

        LLVM.append!(LLVM.incoming(cur),
                     [(cur0, entry), (new_cur, f_next)])
        LLVM.append!(LLVM.incoming(prev_link),
                     [(prev_link0, entry), (new_prev_link, f_next)])
    end

    LLVM.API.LLVMAppendToUsed(LLVM.parent(fn), [fn], 1)
end

# `whisk_gc_alloc(sz) → payload_addr` is the JS-facing export. Thin
# wrapper that forwards to `whisk_gc_alloc_typed(sz, 0)`. JS uses this
# to allocate raw byte buffers (graph blobs, output buffers); the
# layout=0 sentinel tells the mark walker the payload has no
# pointers, skip the scan.
function _emit_gc_alloc_untyped_wrapper!(mod::LLVM.Module)
    T_w = LLVM.Int64Type()

    fn = if haskey(LLVM.functions(mod), "whisk_gc_alloc")
        LLVM.functions(mod)["whisk_gc_alloc"]
    else
        LLVM.Function(mod, "whisk_gc_alloc",
                      LLVM.FunctionType(T_w, [T_w]))
    end
    LLVM.isdeclaration(fn) || return
    LLVM.linkage!(fn, LLVM.API.LLVMExternalLinkage)
    push!(LLVM.function_attributes(fn),
          LLVM.StringAttribute("wasm-export-name", "whisk_gc_alloc"))
    push!(LLVM.function_attributes(fn), LLVM.EnumAttribute("noinline", 0))

    typed_fn = LLVM.functions(mod)["whisk_gc_alloc_typed"]
    sz_param = LLVM.parameters(fn)[1]
    entry = LLVM.BasicBlock(fn, "entry")
    @dispose b=LLVM.IRBuilder() begin
        LLVM.position!(b, entry)
        ret = LLVM.call!(b, LLVM.function_type(typed_fn), typed_fn,
                          [sz_param, LLVM.ConstantInt(T_w, 0)])
        LLVM.tailcall!(ret, true)
        LLVM.ret!(b, ret)
    end
    LLVM.API.LLVMAppendToUsed(LLVM.parent(fn), [fn], 1)
end

# `whisk_gc_reserve(size) → start_addr` — bump-allocate `size` bytes WITHOUT
# adding an outer GC header. Caller fills the bytes with a sequence of
# already-headered objects (i.e. an image). Sweep walks each embedded
# object's header naturally — there is no outer block hiding them.
#
# Use case: `WhiskInstance.uploadImage` copies a host-built image into wasm
# linear memory. The image is a self-describing sequence of GC-headered
# sub-objects. Going through `whisk_gc_alloc` would wrap them all in a
# single outer block whose mark bit is the only thing sweep examines —
# pinning a sub-object inside that block has no effect on whether sweep
# reclaims the outer block. With `whisk_gc_reserve` the embedded headers
# ARE the only headers in the reserved range, and per-object marks /
# sweeps work as expected.
function _emit_gc_reserve!(mod::LLVM.Module, heap_base, heap_next)
    T_w = LLVM.Int64Type()

    fn = if haskey(LLVM.functions(mod), "whisk_gc_reserve")
        LLVM.functions(mod)["whisk_gc_reserve"]
    else
        LLVM.Function(mod, "whisk_gc_reserve",
                      LLVM.FunctionType(T_w, [T_w]))
    end
    LLVM.isdeclaration(fn) || return
    LLVM.linkage!(fn, LLVM.API.LLVMExternalLinkage)
    push!(LLVM.function_attributes(fn),
          LLVM.StringAttribute("wasm-export-name", "whisk_gc_reserve"))
    push!(LLVM.function_attributes(fn), LLVM.EnumAttribute("noinline", 0))

    ensure_fn = LLVM.functions(mod)["whisk_ensure_capacity"]

    sz_param = LLVM.parameters(fn)[1]
    entry = LLVM.BasicBlock(fn, "entry")
    @dispose b=LLVM.IRBuilder() begin
        LLVM.position!(b, entry)
        cur_hn    = LLVM.load!(b, T_w, heap_next)
        is_unset  = LLVM.icmp!(b, LLVM.API.LLVMIntEQ, cur_hn,
                                LLVM.ConstantInt(T_w, 0))
        base_addr = LLVM.ptrtoint!(b, heap_base, T_w)
        actual_hn = LLVM.select!(b, is_unset, base_addr, cur_hn)
        new_hn    = LLVM.add!(b, actual_hn, sz_param)
        LLVM.call!(b, LLVM.function_type(ensure_fn), ensure_fn,
                    LLVM.Value[new_hn])
        LLVM.store!(b, new_hn, heap_next)
        LLVM.ret!(b, actual_hn)
    end
    LLVM.API.LLVMAppendToUsed(LLVM.parent(fn), [fn], 1)
end

@inline function _i64_max!(b::LLVM.IRBuilder, a, c)
    cond = LLVM.icmp!(b, LLVM.API.LLVMIntUGT, a, c)
    return LLVM.select!(b, cond, a, c)
end

# ─────────────────────────────────────────────────────────────────────
# jl_alloc_genericmemory_unchecked — Julia's Memory{T} ctor.
# Memory layout on wasm64 is `{ i64 length, ptr data }` = 16 bytes,
# matching Julia's host-time assumption.
# ─────────────────────────────────────────────────────────────────────

function _emit_gc_alloc_memory!(mod::LLVM.Module)
    T_w   = LLVM.Int64Type()
    T_i8p = LLVM.PointerType(LLVM.Int8Type())

    fn = if haskey(LLVM.functions(mod), "jl_alloc_genericmemory_unchecked")
        LLVM.functions(mod)["jl_alloc_genericmemory_unchecked"]
    else
        ft = LLVM.FunctionType(T_w, [T_w, T_w, T_w])
        LLVM.Function(mod, "jl_alloc_genericmemory_unchecked", ft)
    end
    LLVM.isdeclaration(fn) || return
    LLVM.linkage!(fn, LLVM.API.LLVMInternalLinkage)
    push!(LLVM.function_attributes(fn), LLVM.EnumAttribute("alwaysinline", 0))

    T_ret = LLVM.return_type(LLVM.function_type(fn))
    len_bytes = LLVM.parameters(fn)[2]
    entry = LLVM.BasicBlock(fn, "entry")

    whisk_alloc    = LLVM.functions(mod)["whisk_gc_alloc"]
    whisk_alloc_ft = LLVM.function_type(whisk_alloc)

    @dispose b=LLVM.IRBuilder() begin
        LLVM.position!(b, entry)
        # Memory{T} on wasm64: `{ i64 length, ptr data }` = 16 bytes,
        # followed inline by `len_bytes` of payload data. ONE allocation
        # — splitting it into two `whisk_gc_alloc` calls would create a
        # GC race: if the second alloc triggered mark+sweep, the first
        # allocation isn't yet rooted (its address only lives in an
        # SSA register, not in any GC frame slot), and sweep would
        # reclaim it while still in use. Pack inline so the entire
        # Memory{T} is one heap object that mark sees as a unit.
        total_bytes = LLVM.add!(b, len_bytes, LLVM.ConstantInt(T_w, 16))
        ms_addr     = LLVM.call!(b, whisk_alloc_ft, whisk_alloc, [total_bytes])
        # `data` field = address of inline payload = ms_addr + 16
        data_addr   = LLVM.add!(b, ms_addr, LLVM.ConstantInt(T_w, 16))
        ms_byte_ptr = LLVM.inttoptr!(b, ms_addr, T_i8p)
        slot_ptr    = LLVM.gep!(b, LLVM.Int8Type(), ms_byte_ptr,
                                 [LLVM.ConstantInt(T_w, 8)])
        LLVM.store!(b, data_addr, slot_ptr)
        # Length field at offset 0 — Julia code overwrites it.

        retval = if T_ret isa LLVM.IntegerType
            ms_addr
        else
            cast_ptr0 = LLVM.inttoptr!(b, ms_addr, T_i8p)
            LLVM.addrspacecast!(b, cast_ptr0, T_ret)
        end
        LLVM.ret!(b, retval)
    end
end

# ─────────────────────────────────────────────────────────────────────
# Mark phase: walk pgcstack frames, mark reachable, recurse.
# ─────────────────────────────────────────────────────────────────────

## Precise mark.
##
## `_whisk_mark_obj_at_addr(addr)` is called with a candidate heap
## pointer (a payload-base address — i.e., the value `whisk_gc_alloc*`
## returned for some object). It:
##   1. Bails if `addr` isn't in the heap range.
##   2. Reads the GC header at `[addr-16, addr-8]`. The size word lives
##      at addr-16; the layout-pointer-with-flags at addr-8.
##   3. Bails if the mark bit is already set.
##   4. Sets the mark bit (preserving the layout-pointer bits).
##   5. Decodes layout_ptr; if zero, the payload is raw bytes — no
##      pointers to follow, done.
##   6. Reads `elsz` and `nptrs` from the descriptor. Two cases:
##       * elsz == 0: fixed-size object. For each of `nptrs` offsets,
##         load `*(addr + offset)` and recurse.
##       * elsz != 0: `Memory<T>` with inline elements at `addr + 16`.
##         Read length at `addr + 0`. For each element index `0..len-1`
##         and each per-element offset, load and recurse.
##
## No conservative byte scan, no O(N) heap-walk to find the containing
## object. False-positive mark rate: zero. Mark cost: proportional to
## the actual pointer-graph fan-out, not heap size.
## `_whisk_mark_obj_at_addr(addr)` — "mark and enqueue".
##
## Pre-condition: `addr` is a candidate heap pointer (raw i64).
## Post-condition: if `addr` points to a live, unmarked, non-null
## heap object, that object's MARK bit is set and its address is
## pushed on `_whisk_mark_stack`. The drain function later pops
## addresses and walks their layouts, calling this for each
## pointer-slot value found. No recursion, so deep object graphs
## can't overflow the wasm call stack.
##
## Stack overflow is fatal: we trap with a clear diagnostic so the
## failure mode is loud rather than silent live-data loss.
function _emit_mark_obj_at_addr!(mod, heap_base, heap_next, mark_stack, mark_top)
    T_w  = LLVM.Int64Type()
    T_wp = LLVM.PointerType(T_w)

    fn = if haskey(LLVM.functions(mod), "_whisk_mark_obj_at_addr")
        LLVM.functions(mod)["_whisk_mark_obj_at_addr"]
    else
        LLVM.Function(mod, "_whisk_mark_obj_at_addr",
                      LLVM.FunctionType(LLVM.VoidType(), [T_w]))
    end
    LLVM.isdeclaration(fn) || return
    LLVM.linkage!(fn, LLVM.API.LLVMInternalLinkage)
    push!(LLVM.function_attributes(fn), LLVM.EnumAttribute("noinline", 0))

    addr = LLVM.parameters(fn)[1]
    entry        = LLVM.BasicBlock(fn, "entry")
    in_range     = LLVM.BasicBlock(fn, "in_range")
    do_mark      = LLVM.BasicBlock(fn, "do_mark")
    overflow_blk = LLVM.BasicBlock(fn, "overflow")
    push_blk     = LLVM.BasicBlock(fn, "push")
    done         = LLVM.BasicBlock(fn, "done")

    trap_fn = _get_or_declare_trap!(mod)
    js_fn   = _get_or_declare_js_console!(mod)
    overflow_msg_gv = _intern_message_global!(mod,
        "Whisk GC: mark stack overflow — increase MARK_STACK_CAP")
    overflow_msg_len = sizeof("Whisk GC: mark stack overflow — increase MARK_STACK_CAP")

    @dispose b=LLVM.IRBuilder() begin
        LLVM.position!(b, entry)
        base_addr = LLVM.ptrtoint!(b, heap_base, T_w)
        heap_end  = LLVM.load!(b, T_w, heap_next)
        ge_base   = LLVM.icmp!(b, LLVM.API.LLVMIntUGE, addr, base_addr)
        lt_end    = LLVM.icmp!(b, LLVM.API.LLVMIntULT, addr, heap_end)
        in_heap   = LLVM.and!(b, ge_base, lt_end)
        LLVM.br!(b, in_heap, in_range, done)

        LLVM.position!(b, in_range)
        layout_addr     = LLVM.sub!(b, addr, LLVM.ConstantInt(T_w, 8))
        layout_word_ptr = LLVM.inttoptr!(b, layout_addr, T_wp)
        layout_word     = LLVM.load!(b, T_w, layout_word_ptr)
        marked_bit      = LLVM.and!(b, layout_word,
                                     LLVM.ConstantInt(T_w, FLAG_MARK))
        already         = LLVM.icmp!(b, LLVM.API.LLVMIntNE, marked_bit,
                                      LLVM.ConstantInt(T_w, 0))
        LLVM.br!(b, already, done, do_mark)

        LLVM.position!(b, do_mark)
        new_layout_word = LLVM.or!(b, layout_word,
                                    LLVM.ConstantInt(T_w, FLAG_MARK))
        LLVM.store!(b, new_layout_word, layout_word_ptr)
        # Check stack room before push.
        top_v   = LLVM.load!(b, T_w, mark_top)
        full    = LLVM.icmp!(b, LLVM.API.LLVMIntUGE, top_v,
                              LLVM.ConstantInt(T_w, MARK_STACK_CAP))
        LLVM.br!(b, full, overflow_blk, push_blk)

        LLVM.position!(b, overflow_blk)
        LLVM.call!(b, LLVM.function_type(js_fn), js_fn, LLVM.Value[
            LLVM.ConstantInt(LLVM.Int32Type(), 2),
            overflow_msg_gv,
            LLVM.ConstantInt(LLVM.Int64Type(), overflow_msg_len),
        ])
        LLVM.call!(b, LLVM.function_type(trap_fn), trap_fn, LLVM.Value[])
        LLVM.unreachable!(b)

        LLVM.position!(b, push_blk)
        # *(stack_base + top_v * 8) = addr
        stack_base = LLVM.ptrtoint!(b, mark_stack, T_w)
        slot_byte  = LLVM.shl!(b, top_v, LLVM.ConstantInt(T_w, 3))
        slot_addr  = LLVM.add!(b, stack_base, slot_byte)
        slot_ptr   = LLVM.inttoptr!(b, slot_addr, T_wp)
        LLVM.store!(b, addr, slot_ptr)
        # top++
        new_top = LLVM.add!(b, top_v, LLVM.ConstantInt(T_w, 1))
        LLVM.store!(b, new_top, mark_top)
        LLVM.br!(b, done)

        LLVM.position!(b, done)
        LLVM.ret!(b)
    end
end

## `_whisk_mark_drain()` — pop until empty, walk each layout,
## enqueue children. The work-stack inversion of recursion: same
## visitation order as DFS but stack lives in linear memory.
function _emit_mark_drain!(mod, mark_stack, mark_top)
    T_w  = LLVM.Int64Type()
    T_wp = LLVM.PointerType(T_w)

    fn = if haskey(LLVM.functions(mod), "_whisk_mark_drain")
        LLVM.functions(mod)["_whisk_mark_drain"]
    else
        LLVM.Function(mod, "_whisk_mark_drain",
                      LLVM.FunctionType(LLVM.VoidType(), LLVM.LLVMType[]))
    end
    LLVM.isdeclaration(fn) || return
    LLVM.linkage!(fn, LLVM.API.LLVMInternalLinkage)
    push!(LLVM.function_attributes(fn), LLVM.EnumAttribute("noinline", 0))

    mark_obj = LLVM.functions(mod)["_whisk_mark_obj_at_addr"]
    mark_obj_ft = LLVM.function_type(mark_obj)

    entry        = LLVM.BasicBlock(fn, "entry")
    drain_top    = LLVM.BasicBlock(fn, "drain_top")
    do_pop       = LLVM.BasicBlock(fn, "do_pop")
    has_layout   = LLVM.BasicBlock(fn, "has_layout")
    setup_mem    = LLVM.BasicBlock(fn, "setup_mem")
    setup_fix    = LLVM.BasicBlock(fn, "setup_fix")
    loop_hdr     = LLVM.BasicBlock(fn, "loop_hdr")
    loop_body    = LLVM.BasicBlock(fn, "loop_body")
    next_drain   = LLVM.BasicBlock(fn, "next_drain")
    done         = LLVM.BasicBlock(fn, "done")

    @dispose b=LLVM.IRBuilder() begin
        # The entry block can't have predecessors; the loop-back to
        # drain_top from next_drain would make drain_top a non-entry.
        # Branch unconditionally into drain_top from entry.
        LLVM.position!(b, entry)
        LLVM.br!(b, drain_top)

        LLVM.position!(b, drain_top)
        top_v = LLVM.load!(b, T_w, mark_top)
        empty = LLVM.icmp!(b, LLVM.API.LLVMIntEQ, top_v,
                            LLVM.ConstantInt(T_w, 0))
        LLVM.br!(b, empty, done, do_pop)

        LLVM.position!(b, do_pop)
        new_top    = LLVM.sub!(b, top_v, LLVM.ConstantInt(T_w, 1))
        LLVM.store!(b, new_top, mark_top)
        stack_base = LLVM.ptrtoint!(b, mark_stack, T_w)
        slot_byte  = LLVM.shl!(b, new_top, LLVM.ConstantInt(T_w, 3))
        slot_addr  = LLVM.add!(b, stack_base, slot_byte)
        addr       = LLVM.load!(b, T_w,
                                  LLVM.inttoptr!(b, slot_addr, T_wp))
        # Read layout_ptr from header (low 3 bits are flags; mask off).
        layout_addr     = LLVM.sub!(b, addr, LLVM.ConstantInt(T_w, 8))
        layout_word_ptr = LLVM.inttoptr!(b, layout_addr, T_wp)
        layout_word     = LLVM.load!(b, T_w, layout_word_ptr)
        layout_ptr = LLVM.and!(b, layout_word,
                                LLVM.ConstantInt(T_w, reinterpret(Int64, ~UInt64(0x7))))
        has_lay = LLVM.icmp!(b, LLVM.API.LLVMIntNE, layout_ptr,
                              LLVM.ConstantInt(T_w, 0))
        LLVM.br!(b, has_lay, has_layout, next_drain)

        LLVM.position!(b, has_layout)
        elsz_ptr   = LLVM.inttoptr!(b, layout_ptr, T_wp)
        elsz       = LLVM.load!(b, T_w, elsz_ptr)
        nptrs_addr = LLVM.add!(b, layout_ptr, LLVM.ConstantInt(T_w, 8))
        nptrs_ptr  = LLVM.inttoptr!(b, nptrs_addr, T_wp)
        nptrs      = LLVM.load!(b, T_w, nptrs_ptr)
        offsets_base = LLVM.add!(b, layout_ptr, LLVM.ConstantInt(T_w, 16))
        is_memory  = LLVM.icmp!(b, LLVM.API.LLVMIntNE, elsz,
                                 LLVM.ConstantInt(T_w, 0))
        LLVM.br!(b, is_memory, setup_mem, setup_fix)

        LLVM.position!(b, setup_fix)
        LLVM.br!(b, loop_hdr)

        LLVM.position!(b, setup_mem)
        len_ptr   = LLVM.inttoptr!(b, addr, T_wp)
        len_v     = LLVM.load!(b, T_w, len_ptr)
        iters_mem = LLVM.mul!(b, len_v, nptrs)
        data_base = LLVM.add!(b, addr, LLVM.ConstantInt(T_w, 16))
        LLVM.br!(b, loop_hdr)

        LLVM.position!(b, loop_hdr)
        i      = LLVM.phi!(b, T_w)
        iters  = LLVM.phi!(b, T_w)
        base_v = LLVM.phi!(b, T_w)
        in_iter = LLVM.icmp!(b, LLVM.API.LLVMIntULT, i, iters)
        LLVM.br!(b, in_iter, loop_body, next_drain)

        LLVM.position!(b, loop_body)
        ptr_idx  = LLVM.urem!(b, i, nptrs)
        elem_idx = LLVM.udiv!(b, i, nptrs)
        ptr_idx_x8 = LLVM.shl!(b, ptr_idx, LLVM.ConstantInt(T_w, 3))
        off_addr   = LLVM.add!(b, offsets_base, ptr_idx_x8)
        off_v      = LLVM.load!(b, T_w,
                                 LLVM.inttoptr!(b, off_addr, T_wp))
        elem_off   = LLVM.mul!(b, elem_idx, elsz)
        slot_addr2 = LLVM.add!(b, base_v, elem_off)
        slot_addr2 = LLVM.add!(b, slot_addr2, off_v)
        slot_val   = LLVM.load!(b, T_w,
                                 LLVM.inttoptr!(b, slot_addr2, T_wp))
        LLVM.call!(b, mark_obj_ft, mark_obj, [slot_val])
        i_next = LLVM.add!(b, i, LLVM.ConstantInt(T_w, 1))
        LLVM.br!(b, loop_hdr)

        LLVM.position!(b, next_drain)
        LLVM.br!(b, drain_top)

        LLVM.position!(b, done)
        LLVM.ret!(b)

        LLVM.append!(LLVM.incoming(i),
                     [(LLVM.ConstantInt(T_w, 0), setup_fix),
                      (LLVM.ConstantInt(T_w, 0), setup_mem),
                      (i_next, loop_body)])
        LLVM.append!(LLVM.incoming(iters),
                     [(nptrs,     setup_fix),
                      (iters_mem, setup_mem),
                      (iters,     loop_body)])
        LLVM.append!(LLVM.incoming(base_v),
                     [(addr,      setup_fix),
                      (data_base, setup_mem),
                      (base_v,    loop_body)])
    end
end

function _emit_gc_mark!(mod, pgcstack, js_roots, js_roots_len, mark_top)
    T_w  = LLVM.Int64Type()
    T_wp = LLVM.PointerType(T_w)

    fn = if haskey(LLVM.functions(mod), "_whisk_gc_mark")
        LLVM.functions(mod)["_whisk_gc_mark"]
    else
        LLVM.Function(mod, "_whisk_gc_mark",
                      LLVM.FunctionType(LLVM.VoidType(), LLVM.LLVMType[]))
    end
    LLVM.isdeclaration(fn) || return
    LLVM.linkage!(fn, LLVM.API.LLVMInternalLinkage)
    push!(LLVM.function_attributes(fn), LLVM.EnumAttribute("noinline", 0))

    mark_obj    = LLVM.functions(mod)["_whisk_mark_obj_at_addr"]
    mark_obj_ft = LLVM.function_type(mark_obj)
    mark_drain  = LLVM.functions(mod)["_whisk_mark_drain"]

    entry          = LLVM.BasicBlock(fn, "entry")
    frame_loop     = LLVM.BasicBlock(fn, "frame_loop")
    frame_check    = LLVM.BasicBlock(fn, "frame_check")
    slot_loop      = LLVM.BasicBlock(fn, "slot_loop")
    slot_body      = LLVM.BasicBlock(fn, "slot_body")
    next_frame     = LLVM.BasicBlock(fn, "next_frame")
    pgcstack_done  = LLVM.BasicBlock(fn, "pgcstack_done")
    roots_loop     = LLVM.BasicBlock(fn, "roots_loop")
    roots_body     = LLVM.BasicBlock(fn, "roots_body")
    done           = LLVM.BasicBlock(fn, "done")

    @dispose b=LLVM.IRBuilder() begin
        LLVM.position!(b, entry)
        # Reset the mark stack at the start of each collection.
        LLVM.store!(b, LLVM.ConstantInt(T_w, 0), mark_top)
        cur_frame0 = LLVM.load!(b, T_w, pgcstack)
        LLVM.br!(b, frame_loop)

        LLVM.position!(b, frame_loop)
        cur_frame = LLVM.phi!(b, T_w)
        is_zero = LLVM.icmp!(b, LLVM.API.LLVMIntEQ, cur_frame,
                              LLVM.ConstantInt(T_w, 0))
        LLVM.br!(b, is_zero, pgcstack_done, frame_check)

        LLVM.position!(b, frame_check)
        n_addr  = LLVM.add!(b, cur_frame, LLVM.ConstantInt(T_w, 8))
        n_slots = LLVM.load!(b, T_w, LLVM.inttoptr!(b, n_addr, T_wp))
        LLVM.br!(b, slot_loop)

        LLVM.position!(b, slot_loop)
        i = LLVM.phi!(b, T_w)
        in_range = LLVM.icmp!(b, LLVM.API.LLVMIntULT, i, n_slots)
        LLVM.br!(b, in_range, slot_body, next_frame)

        LLVM.position!(b, slot_body)
        i_x8        = LLVM.shl!(b, i, LLVM.ConstantInt(T_w, 3))
        slot_offset = LLVM.add!(b, i_x8, LLVM.ConstantInt(T_w, 16))
        slot_addr   = LLVM.add!(b, cur_frame, slot_offset)
        slot_value  = LLVM.load!(b, T_w, LLVM.inttoptr!(b, slot_addr, T_wp))
        LLVM.call!(b, mark_obj_ft, mark_obj, [slot_value])
        i_next = LLVM.add!(b, i, LLVM.ConstantInt(T_w, 1))
        LLVM.br!(b, slot_loop)

        LLVM.position!(b, next_frame)
        prev_frame = LLVM.load!(b, T_w, LLVM.inttoptr!(b, cur_frame, T_wp))
        LLVM.br!(b, frame_loop)

        # ── pgcstack walk done; now walk the JS-roots table.
        #
        # `whisk_pin` only writes slots 0..len; slots len..cap are
        # kept zero, so a conservative scan would also work, but
        # walking exactly `len` entries keeps the algorithm precise
        # and bounded.
        LLVM.position!(b, pgcstack_done)
        roots_len_v  = LLVM.load!(b, T_w, js_roots_len)
        roots_base   = LLVM.ptrtoint!(b, js_roots, T_w)
        LLVM.br!(b, roots_loop)

        LLVM.position!(b, roots_loop)
        rj = LLVM.phi!(b, T_w)
        rj_in = LLVM.icmp!(b, LLVM.API.LLVMIntULT, rj, roots_len_v)
        LLVM.br!(b, rj_in, roots_body, done)

        LLVM.position!(b, roots_body)
        rj_x8       = LLVM.shl!(b, rj, LLVM.ConstantInt(T_w, 3))
        root_addr   = LLVM.add!(b, roots_base, rj_x8)
        root_value  = LLVM.load!(b, T_w, LLVM.inttoptr!(b, root_addr, T_wp))
        LLVM.call!(b, mark_obj_ft, mark_obj, [root_value])
        rj_next = LLVM.add!(b, rj, LLVM.ConstantInt(T_w, 1))
        LLVM.br!(b, roots_loop)

        LLVM.position!(b, done)
        # All roots pushed; drain the mark stack to do the actual graph
        # walk. No recursion, so deep object graphs can't overflow the
        # wasm call stack.
        LLVM.call!(b, LLVM.function_type(mark_drain), mark_drain, LLVM.Value[])
        LLVM.ret!(b)

        LLVM.append!(LLVM.incoming(cur_frame),
                     [(cur_frame0, entry), (prev_frame, next_frame)])
        LLVM.append!(LLVM.incoming(i),
                     [(LLVM.ConstantInt(T_w, 0), frame_check),
                      (i_next, slot_body)])
        LLVM.append!(LLVM.incoming(rj),
                     [(LLVM.ConstantInt(T_w, 0), pgcstack_done),
                      (rj_next, roots_body)])
    end
end

# ─────────────────────────────────────────────────────────────────────
# Sweep phase
# ─────────────────────────────────────────────────────────────────────

## Sweep walks the heap linearly from `__heap_base`, advancing by
## the size field at each object's offset 0. The layout-pointer-with-
## flags word at offset 8 is rewritten:
##   * marked alive → clear MARK bit, leave layout_ptr intact.
##   * unmarked → push onto free-list and set FLAG_FREE. The free-list
##     link goes into the payload's first 8 bytes; the offset-8 word
##     becomes `0 | FLAG_FREE` (no layout — block is dead).
## Sweep with free-list coalescing + bump-pointer reset.
##
## Walks the heap linearly. Tracks "current run of dead blocks"
## (run_start, run_size) as loop phi state. State transitions:
##   * live block + run in progress → flush run to free-list, clear mark
##   * live block + no run           → clear mark
##   * dead block + run in progress → extend run (run_size += size)
##   * dead block + no run           → start run (run_start = obj, run_size = size)
##
## After the walk, if a run is still open:
##   * run reaches heap_end → reset _whisk_heap_next to run_start.
##     The bump pointer steps back, freeing the entire tail region for
##     fresh allocations without bumping past the old high-water.
##   * run is in the middle  → flush as a single coalesced free block.
##
## A run's flushed block has:
##   * size field (offset 0) = total run bytes (sum of merged blocks)
##   * layout/flags (offset 8) = FLAG_FREE
##   * payload[0]               = previous free-list head
##   * free_list = run_start
function _emit_gc_sweep!(mod, heap_base, heap_next, free_list)
    T_w  = LLVM.Int64Type()
    T_wp = LLVM.PointerType(T_w)

    fn = if haskey(LLVM.functions(mod), "_whisk_gc_sweep")
        LLVM.functions(mod)["_whisk_gc_sweep"]
    else
        LLVM.Function(mod, "_whisk_gc_sweep",
                      LLVM.FunctionType(LLVM.VoidType(), LLVM.LLVMType[]))
    end
    LLVM.isdeclaration(fn) || return
    LLVM.linkage!(fn, LLVM.API.LLVMInternalLinkage)
    push!(LLVM.function_attributes(fn), LLVM.EnumAttribute("noinline", 0))

    entry          = LLVM.BasicBlock(fn, "entry")
    obj_loop       = LLVM.BasicBlock(fn, "obj_loop")
    obj_check      = LLVM.BasicBlock(fn, "obj_check")
    live_branch    = LLVM.BasicBlock(fn, "live_branch")
    live_flush     = LLVM.BasicBlock(fn, "live_flush")
    live_no_flush  = LLVM.BasicBlock(fn, "live_no_flush")
    live_end       = LLVM.BasicBlock(fn, "live_end")
    dead_branch    = LLVM.BasicBlock(fn, "dead_branch")
    dead_extend    = LLVM.BasicBlock(fn, "dead_extend")
    dead_start     = LLVM.BasicBlock(fn, "dead_start")
    dead_end       = LLVM.BasicBlock(fn, "dead_end")
    after_loop     = LLVM.BasicBlock(fn, "after_loop")
    final_open     = LLVM.BasicBlock(fn, "final_open")
    bump_reset     = LLVM.BasicBlock(fn, "bump_reset")
    final_flush    = LLVM.BasicBlock(fn, "final_flush")
    store_live     = LLVM.BasicBlock(fn, "store_live")

    live_gv = LLVM.globals(LLVM.parent(fn))["_whisk_gc_live_bytes"]

    # Inline helper: emit a "flush run to free-list" sequence.
    function flush_run!(b, run_start, run_size)
        # *(run_start) = run_size  (overwrites first block's size)
        size_ptr = LLVM.inttoptr!(b, run_start, T_wp)
        LLVM.store!(b, run_size, size_ptr)
        # *(run_start + 8) = FLAG_FREE
        flags_addr = LLVM.add!(b, run_start, LLVM.ConstantInt(T_w, HEADER_FLAGS_OFF))
        flags_ptr  = LLVM.inttoptr!(b, flags_addr, T_wp)
        LLVM.store!(b, LLVM.ConstantInt(T_w, FLAG_FREE), flags_ptr)
        # payload[0] = old free_list head; free_list = run_start
        payload_addr = LLVM.add!(b, run_start, LLVM.ConstantInt(T_w, HEADER_SIZE))
        payload_ptr  = LLVM.inttoptr!(b, payload_addr, T_wp)
        old_head     = LLVM.load!(b, T_w, free_list)
        LLVM.store!(b, old_head, payload_ptr)
        LLVM.store!(b, run_start, free_list)
        return nothing
    end

    @dispose b=LLVM.IRBuilder() begin
        LLVM.position!(b, entry)
        # Reset the free-list head; we'll rebuild it from coalesced
        # runs as we walk the heap.
        LLVM.store!(b, LLVM.ConstantInt(T_w, 0), free_list)
        base_addr = LLVM.ptrtoint!(b, heap_base, T_w)
        heap_end  = LLVM.load!(b, T_w, heap_next)
        LLVM.br!(b, obj_loop)

        # ── obj_loop: phi-merge of obj, live, run_start, run_size.
        LLVM.position!(b, obj_loop)
        obj       = LLVM.phi!(b, T_w)
        live      = LLVM.phi!(b, T_w)
        run_start = LLVM.phi!(b, T_w)   # 0 = no run open
        run_size  = LLVM.phi!(b, T_w)
        past = LLVM.icmp!(b, LLVM.API.LLVMIntUGE, obj, heap_end)
        LLVM.br!(b, past, after_loop, obj_check)

        LLVM.position!(b, obj_check)
        size_v        = LLVM.load!(b, T_w, LLVM.inttoptr!(b, obj, T_wp))
        layout_addr   = LLVM.add!(b, obj, LLVM.ConstantInt(T_w, HEADER_FLAGS_OFF))
        layout_ptr_lo = LLVM.inttoptr!(b, layout_addr, T_wp)
        layout_word   = LLVM.load!(b, T_w, layout_ptr_lo)
        marked        = LLVM.and!(b, layout_word, LLVM.ConstantInt(T_w, FLAG_MARK))
        was_marked    = LLVM.icmp!(b, LLVM.API.LLVMIntNE, marked,
                                    LLVM.ConstantInt(T_w, 0))
        LLVM.br!(b, was_marked, live_branch, dead_branch)

        # ── Live block: maybe flush an open run, then clear MARK.
        LLVM.position!(b, live_branch)
        has_run_live = LLVM.icmp!(b, LLVM.API.LLVMIntNE, run_start,
                                    LLVM.ConstantInt(T_w, 0))
        LLVM.br!(b, has_run_live, live_flush, live_no_flush)

        LLVM.position!(b, live_flush)
        flush_run!(b, run_start, run_size)
        LLVM.br!(b, live_no_flush)

        LLVM.position!(b, live_no_flush)
        # Clear MARK bit, preserve layout_ptr + other flags.
        cleared = LLVM.and!(b, layout_word,
                            LLVM.ConstantInt(T_w, reinterpret(Int64, ~FLAG_MARK)))
        LLVM.store!(b, cleared, layout_ptr_lo)
        LLVM.br!(b, live_end)

        LLVM.position!(b, live_end)
        next_obj_live = LLVM.add!(b, obj, size_v)
        next_live_v   = LLVM.add!(b, live, size_v)
        # Run is closed (whether it was open or not); next iteration
        # starts with run_start=0, run_size=0.
        LLVM.br!(b, obj_loop)

        # ── Dead block: extend or start a run.
        LLVM.position!(b, dead_branch)
        has_run_dead = LLVM.icmp!(b, LLVM.API.LLVMIntNE, run_start,
                                    LLVM.ConstantInt(T_w, 0))
        LLVM.br!(b, has_run_dead, dead_extend, dead_start)

        LLVM.position!(b, dead_extend)
        extended_size = LLVM.add!(b, run_size, size_v)
        LLVM.br!(b, dead_end)

        LLVM.position!(b, dead_start)
        # Open new run starting at obj.
        LLVM.br!(b, dead_end)

        LLVM.position!(b, dead_end)
        # next run state
        next_run_start = LLVM.phi!(b, T_w)
        next_run_size  = LLVM.phi!(b, T_w)
        next_obj_dead  = LLVM.add!(b, obj, size_v)
        LLVM.br!(b, obj_loop)

        # ── after_loop: check if a run is still open.
        LLVM.position!(b, after_loop)
        run_open = LLVM.icmp!(b, LLVM.API.LLVMIntNE, run_start,
                                LLVM.ConstantInt(T_w, 0))
        LLVM.br!(b, run_open, final_open, store_live)

        LLVM.position!(b, final_open)
        # If the run reaches heap_end, prefer bump-reset (cheaper than
        # free-list reuse — keeps heap small AND coalesced). Otherwise
        # flush as one big free block.
        run_end_addr = LLVM.add!(b, run_start, run_size)
        at_tail      = LLVM.icmp!(b, LLVM.API.LLVMIntEQ, run_end_addr, heap_end)
        LLVM.br!(b, at_tail, bump_reset, final_flush)

        LLVM.position!(b, bump_reset)
        # Pull heap_next back to the start of the trailing dead run.
        # This is the wasm-equivalent of returning memory to the
        # allocator (we can't actually shrink wasm linear memory, but
        # we CAN avoid pushing the bump pointer further next time).
        LLVM.store!(b, run_start, heap_next)
        LLVM.br!(b, store_live)

        LLVM.position!(b, final_flush)
        flush_run!(b, run_start, run_size)
        LLVM.br!(b, store_live)

        LLVM.position!(b, store_live)
        LLVM.store!(b, live, live_gv)
        LLVM.ret!(b)

        # Wire up phis.
        LLVM.append!(LLVM.incoming(obj),
                     [(base_addr,     entry),
                      (next_obj_live, live_end),
                      (next_obj_dead, dead_end)])
        LLVM.append!(LLVM.incoming(live),
                     [(LLVM.ConstantInt(T_w, 0), entry),
                      (next_live_v,              live_end),
                      (live,                     dead_end)])
        LLVM.append!(LLVM.incoming(run_start),
                     [(LLVM.ConstantInt(T_w, 0), entry),
                      (LLVM.ConstantInt(T_w, 0), live_end),
                      (next_run_start,           dead_end)])
        LLVM.append!(LLVM.incoming(run_size),
                     [(LLVM.ConstantInt(T_w, 0), entry),
                      (LLVM.ConstantInt(T_w, 0), live_end),
                      (next_run_size,            dead_end)])
        LLVM.append!(LLVM.incoming(next_run_start),
                     [(run_start, dead_extend),
                      (obj,       dead_start)])
        LLVM.append!(LLVM.incoming(next_run_size),
                     [(extended_size, dead_extend),
                      (size_v,        dead_start)])
    end
end

function _emit_gc_collect!(mod::LLVM.Module)
    fn = if haskey(LLVM.functions(mod), "whisk_gc_collect")
        LLVM.functions(mod)["whisk_gc_collect"]
    else
        LLVM.Function(mod, "whisk_gc_collect",
                      LLVM.FunctionType(LLVM.VoidType(), LLVM.LLVMType[]))
    end
    LLVM.isdeclaration(fn) || return
    # External linkage + wasm-export-name attribute exports the symbol
    # at link time. Append to `@llvm.used` so the optimizer doesn't DCE
    # it before the linker sees it.
    LLVM.linkage!(fn, LLVM.API.LLVMExternalLinkage)
    push!(LLVM.function_attributes(fn),
          LLVM.StringAttribute("wasm-export-name", "whisk_gc_collect"))
    push!(LLVM.function_attributes(fn), LLVM.EnumAttribute("noinline", 0))

    mark_fn  = LLVM.functions(mod)["_whisk_gc_mark"]
    sweep_fn = LLVM.functions(mod)["_whisk_gc_sweep"]
    nc_gv    = LLVM.globals(mod)["_whisk_gc_num_collections"]
    T_w      = LLVM.Int64Type()
    entry    = LLVM.BasicBlock(fn, "entry")

    @dispose b=LLVM.IRBuilder() begin
        LLVM.position!(b, entry)
        LLVM.call!(b, LLVM.function_type(mark_fn),  mark_fn,  LLVM.Value[])
        LLVM.call!(b, LLVM.function_type(sweep_fn), sweep_fn, LLVM.Value[])
        nc_old = LLVM.load!(b, T_w, nc_gv)
        nc_new = LLVM.add!(b, nc_old, LLVM.ConstantInt(T_w, 1))
        LLVM.store!(b, nc_new, nc_gv)
        LLVM.ret!(b)
    end

    mod_obj = LLVM.parent(fn)
    LLVM.API.LLVMAppendToUsed(mod_obj, [fn], 1)
end

# ─────────────────────────────────────────────────────────────────────
# JS-roots pin/unpin
# ─────────────────────────────────────────────────────────────────────
#
# `whisk_pin(ptr)` and `whisk_unpin(ptr)` are wasm-exported. JS calls
# `whisk_pin` after allocating a long-lived buffer via `whisk_gc_alloc`
# so the buffer survives mark+sweep across calls. Pin order doesn't
# matter; unpin uses linear scan + swap-with-last to keep the table
# dense at offsets 0..len.
#
# Overflow is a hard trap (with diagnostic) rather than a soft return:
# the caller mis-coding pin/unpin balance is much more dangerous than
# letting it know loudly.

function _emit_js_pin!(mod::LLVM.Module, js_roots, js_roots_len)
    T_w  = LLVM.Int64Type()
    T_wp = LLVM.PointerType(T_w)

    fn = if haskey(LLVM.functions(mod), "whisk_pin")
        LLVM.functions(mod)["whisk_pin"]
    else
        LLVM.Function(mod, "whisk_pin",
                      LLVM.FunctionType(LLVM.VoidType(), [T_w]))
    end
    LLVM.isdeclaration(fn) || return
    LLVM.linkage!(fn, LLVM.API.LLVMExternalLinkage)
    push!(LLVM.function_attributes(fn),
          LLVM.StringAttribute("wasm-export-name", "whisk_pin"))
    push!(LLVM.function_attributes(fn), LLVM.EnumAttribute("noinline", 0))

    ptr_arg = LLVM.parameters(fn)[1]
    entry      = LLVM.BasicBlock(fn, "entry")
    overflow   = LLVM.BasicBlock(fn, "overflow")
    do_push    = LLVM.BasicBlock(fn, "do_push")

    msg     = "whisk_pin: JS-roots table overflowed (cap = $(JS_ROOTS_CAP))"
    msg_gv  = _intern_message_global!(mod, msg)
    msg_len = sizeof(msg)
    js_fn   = _get_or_declare_js_console!(mod)
    trap_fn = _get_or_declare_trap!(mod)

    @dispose b=LLVM.IRBuilder() begin
        LLVM.position!(b, entry)
        cur_len = LLVM.load!(b, T_w, js_roots_len)
        full = LLVM.icmp!(b, LLVM.API.LLVMIntUGE, cur_len,
                            LLVM.ConstantInt(T_w, JS_ROOTS_CAP))
        LLVM.br!(b, full, overflow, do_push)

        LLVM.position!(b, overflow)
        LLVM.call!(b, LLVM.function_type(js_fn), js_fn,
                    [LLVM.ConstantInt(LLVM.Int32Type(), 2),
                     msg_gv,
                     LLVM.ConstantInt(T_w, msg_len)])
        LLVM.call!(b, LLVM.function_type(trap_fn), trap_fn, LLVM.Value[])
        LLVM.unreachable!(b)

        LLVM.position!(b, do_push)
        roots_base = LLVM.ptrtoint!(b, js_roots, T_w)
        offset     = LLVM.shl!(b, cur_len, LLVM.ConstantInt(T_w, 3))
        slot_addr  = LLVM.add!(b, roots_base, offset)
        LLVM.store!(b, ptr_arg, LLVM.inttoptr!(b, slot_addr, T_wp))
        new_len    = LLVM.add!(b, cur_len, LLVM.ConstantInt(T_w, 1))
        LLVM.store!(b, new_len, js_roots_len)
        LLVM.ret!(b)
    end

    LLVM.API.LLVMAppendToUsed(LLVM.parent(fn), [fn], 1)
end

function _emit_js_unpin!(mod::LLVM.Module, js_roots, js_roots_len)
    T_w  = LLVM.Int64Type()
    T_wp = LLVM.PointerType(T_w)

    fn = if haskey(LLVM.functions(mod), "whisk_unpin")
        LLVM.functions(mod)["whisk_unpin"]
    else
        LLVM.Function(mod, "whisk_unpin",
                      LLVM.FunctionType(LLVM.VoidType(), [T_w]))
    end
    LLVM.isdeclaration(fn) || return
    LLVM.linkage!(fn, LLVM.API.LLVMExternalLinkage)
    push!(LLVM.function_attributes(fn),
          LLVM.StringAttribute("wasm-export-name", "whisk_unpin"))
    push!(LLVM.function_attributes(fn), LLVM.EnumAttribute("noinline", 0))

    ptr_arg = LLVM.parameters(fn)[1]
    entry      = LLVM.BasicBlock(fn, "entry")
    loop_check = LLVM.BasicBlock(fn, "loop_check")
    loop_body  = LLVM.BasicBlock(fn, "loop_body")
    advance    = LLVM.BasicBlock(fn, "advance")
    swap       = LLVM.BasicBlock(fn, "swap")
    done       = LLVM.BasicBlock(fn, "done")

    @dispose b=LLVM.IRBuilder() begin
        LLVM.position!(b, entry)
        cur_len    = LLVM.load!(b, T_w, js_roots_len)
        roots_base = LLVM.ptrtoint!(b, js_roots, T_w)
        LLVM.br!(b, loop_check)

        LLVM.position!(b, loop_check)
        i = LLVM.phi!(b, T_w)
        in_range = LLVM.icmp!(b, LLVM.API.LLVMIntULT, i, cur_len)
        LLVM.br!(b, in_range, loop_body, done)

        LLVM.position!(b, loop_body)
        offset    = LLVM.shl!(b, i, LLVM.ConstantInt(T_w, 3))
        slot_addr = LLVM.add!(b, roots_base, offset)
        slot_p    = LLVM.inttoptr!(b, slot_addr, T_wp)
        v         = LLVM.load!(b, T_w, slot_p)
        match     = LLVM.icmp!(b, LLVM.API.LLVMIntEQ, v, ptr_arg)
        LLVM.br!(b, match, swap, advance)

        LLVM.position!(b, advance)
        i_next = LLVM.add!(b, i, LLVM.ConstantInt(T_w, 1))
        LLVM.br!(b, loop_check)

        # Found at slot i. Swap with the last live slot, zero the
        # vacated tail slot (so the conservative scanner can't keep
        # the formerly-pinned object alive via stale bytes), and
        # decrement len.
        LLVM.position!(b, swap)
        last_i      = LLVM.sub!(b, cur_len, LLVM.ConstantInt(T_w, 1))
        last_offset = LLVM.shl!(b, last_i, LLVM.ConstantInt(T_w, 3))
        last_addr   = LLVM.add!(b, roots_base, last_offset)
        last_p      = LLVM.inttoptr!(b, last_addr, T_wp)
        last_v      = LLVM.load!(b, T_w, last_p)
        LLVM.store!(b, last_v, slot_p)
        LLVM.store!(b, LLVM.ConstantInt(T_w, 0), last_p)
        LLVM.store!(b, last_i, js_roots_len)
        LLVM.br!(b, done)

        LLVM.position!(b, done)
        LLVM.ret!(b)

        LLVM.append!(LLVM.incoming(i),
                     [(LLVM.ConstantInt(T_w, 0), entry),
                      (i_next, advance)])
    end

    LLVM.API.LLVMAppendToUsed(LLVM.parent(fn), [fn], 1)
end

# ─────────────────────────────────────────────────────────────────────
# Lowering passes invoked from finish_module!
# ─────────────────────────────────────────────────────────────────────

"""
    lower_gc_alloc_wasm!(mod::LLVM.Module)

Replace every `julia.gc_alloc_obj(_, sz_i64, ty)` with
`whisk_gc_alloc_typed(sz_i64, layout_ptr)` plus addrspacecast.
The type pointer is decoded at compile time and a per-type layout
descriptor is emitted as wasm rodata. The descriptor's address
becomes the layout_ptr stored in the GC header — the precise mark
walker reads it to find the object's actual heap-pointer slots.

Falls back to `layout_ptr = 0` (mark walker treats payload as raw
bytes) if the type pointer can't be decoded — in practice that
happens for sub-compiled helpers where Julia codegen used a different
shape; those sites won't carry pointers across calls anyway.
"""
function lower_gc_alloc_wasm!(mod::LLVM.Module)
    haskey(LLVM.functions(mod), "julia.gc_alloc_obj") || return
    alloc_obj = LLVM.functions(mod)["julia.gc_alloc_obj"]
    T_ret = LLVM.return_type(LLVM.function_type(alloc_obj))
    T_w = LLVM.Int64Type()

    typed_alloc    = LLVM.functions(mod)["whisk_gc_alloc_typed"]
    typed_alloc_ft = LLVM.function_type(typed_alloc)
    T_ptr0 = LLVM.PointerType(LLVM.Int8Type())

    for use in collect(LLVM.uses(alloc_obj))
        call = LLVM.user(use)::LLVM.CallInst
        ops  = LLVM.arguments(call)
        sz   = ops[2]
        ty   = ops[3]
        layout_val = _layout_value_for_type_operand!(mod, ty, false)
        @dispose builder=LLVM.IRBuilder() begin
            LLVM.position!(builder, call)
            i64_addr = LLVM.call!(builder, typed_alloc_ft, typed_alloc,
                                   [sz, layout_val])
            as_ptr0  = LLVM.inttoptr!(builder, i64_addr, T_ptr0)
            as_ptr   = LLVM.addrspacecast!(builder, as_ptr0, T_ret)
            LLVM.replace_uses!(call, as_ptr)
        end
        LLVM.erase!(call)
    end
    isempty(LLVM.uses(alloc_obj)) && LLVM.erase!(alloc_obj)
end

# Decode the type-pointer LLVM operand at a julia.gc_alloc_obj or
# jl_alloc_genericmemory_unchecked call site to its Julia DataType.
# Two shapes appear in practice:
#   1. `inttoptr (i64 N to ptr ...)` ConstantExpr — frontend output.
#   2. `load ptr, ptr @jl_global_NNN` — Julia's `julia.call2` lowering.
# Returns the Julia DataType, or `nothing` if neither shape decodes.
function _decode_datatype_operand(operand)
    if operand isa LLVM.LoadInst
        ptr_op = LLVM.operands(operand)[1]
        ptr_op isa LLVM.GlobalVariable || return nothing
        init = LLVM.initializer(ptr_op)
        init === nothing && return nothing
        return _decode_datatype_operand(init)
    end
    if operand isa LLVM.ConstantExpr
        LLVM.opcode(operand) == LLVM.API.LLVMIntToPtr || return nothing
        addr_op = LLVM.operands(operand)[1]
        addr_op isa LLVM.ConstantInt || return nothing
        addr = convert(Int, addr_op)
        addr <= 0x100000 && return nothing
        addr % 8 != 0    && return nothing
        obj = try
            Base.unsafe_pointer_to_objref(Ptr{Nothing}(addr))
        catch
            return nothing
        end
        return obj isa DataType ? obj : nothing
    end
    return nothing
end

# Resolve the LLVM i64 layout-pointer value for a type-operand at an
# alloc call site. Returns either:
#   * a `ptrtoint` of a freshly emitted descriptor global (precise), or
#   * `i64 0` (raw bytes — used when type can't be decoded or for an
#     element type with no pointer fields, where scanning is a no-op).
#
# `as_memory` controls Memory<T>-vs-fixed-object semantics:
#   * `false`: descriptor describes a fixed-size object; offsets are
#     within the payload at addr+0.
#   * `true` : descriptor describes a `Memory<T>`; `elsz` and per-element
#     offsets are computed from `T = ty.parameters[1]`.
function _layout_value_for_type_operand!(mod::LLVM.Module, operand, as_memory::Bool)
    T_w = LLVM.Int64Type()
    ty = _decode_datatype_operand(operand)
    if ty === nothing
        # Type couldn't be decoded — codegen shape we don't recognise.
        # Track for end-of-compile summary; the fallback's correctness
        # depends on the alloc site not actually containing live heap
        # pointers (sub-compiled helpers' short-lived boxes typically
        # don't, but it's not a guarantee).
        _record_layout_fallback!(mod, operand, as_memory)
        return LLVM.ConstantInt(T_w, 0)
    end
    elsz, offsets = if as_memory
        memory_element_layout(ty)
    else
        # Fixed-size object: elsz = 0 (sentinel), offsets within payload.
        if isbitstype(ty) || Base.datatype_pointerfree(ty)
            return LLVM.ConstantInt(T_w, 0)
        end
        (UInt64(0), pointer_offsets(ty))
    end
    if isempty(offsets)
        # Pointer-free element (e.g. Memory{Float32}, bits Tuple). No
        # warning: the mark walker has nothing useful to do with the
        # bytes and layout=0 makes that explicit + a touch faster.
        return LLVM.ConstantInt(T_w, 0)
    end
    descriptor = get_or_emit_layout_descriptor!(mod, elsz, offsets)
    return LLVM.const_ptrtoint(descriptor, T_w)
end

"""
    lower_alloc_genericmemory!(mod::LLVM.Module)

Rewrite every `jl_alloc_genericmemory_unchecked(ty, len_bytes, owner)`
call into a precise allocation:

    addr = whisk_gc_alloc_typed(len_bytes + 16, layout_ptr_for_Memory<T>)
    *(addr + 8) = addr + 16    // data ptr field
    return addr

`ty` is decoded at compile time to recover `T`, and the per-element
layout (elsz, pointer offsets within each T element) is baked into a
wasm rodata descriptor. Falls back to a layout=0 (raw bytes) call if
`ty` can't be decoded.
"""
function lower_alloc_genericmemory!(mod::LLVM.Module)
    haskey(LLVM.functions(mod), "jl_alloc_genericmemory_unchecked") || return
    alloc_mem = LLVM.functions(mod)["jl_alloc_genericmemory_unchecked"]
    T_w  = LLVM.Int64Type()
    T_wp = LLVM.PointerType(T_w)
    T_ret = LLVM.return_type(LLVM.function_type(alloc_mem))

    typed_alloc    = LLVM.functions(mod)["whisk_gc_alloc_typed"]
    typed_alloc_ft = LLVM.function_type(typed_alloc)

    for use in collect(LLVM.uses(alloc_mem))
        call = LLVM.user(use)
        call isa LLVM.CallInst || continue
        ops = LLVM.arguments(call)
        length(ops) >= 2 || continue
        ty_operand = ops[1]
        len_bytes  = ops[2]
        layout_val = _layout_value_for_type_operand!(mod, ty_operand, true)
        @dispose builder=LLVM.IRBuilder() begin
            LLVM.position!(builder, call)
            total_bytes = LLVM.add!(builder, len_bytes,
                                     LLVM.ConstantInt(T_w, 16))
            ms_addr = LLVM.call!(builder, typed_alloc_ft, typed_alloc,
                                  [total_bytes, layout_val])
            # data ptr field at offset 8 = ms_addr + 16
            data_addr = LLVM.add!(builder, ms_addr, LLVM.ConstantInt(T_w, 16))
            slot_addr = LLVM.add!(builder, ms_addr, LLVM.ConstantInt(T_w, 8))
            LLVM.store!(builder, data_addr,
                         LLVM.inttoptr!(builder, slot_addr, T_wp))
            # Return type may be ptr or i64 depending on Julia codegen;
            # convert as needed.
            ret_val = if T_ret isa LLVM.IntegerType
                ms_addr
            else
                ms_byte_ptr = LLVM.inttoptr!(builder, ms_addr,
                                              LLVM.PointerType(LLVM.Int8Type()))
                LLVM.addrspacecast!(builder, ms_byte_ptr, T_ret)
            end
            LLVM.replace_uses!(call, ret_val)
        end
        LLVM.erase!(call)
    end
end

"""
    lower_gc_frame_intrinsics!(mod::LLVM.Module)

Lower Julia 1.11+ GC-frame intrinsics into linked-list operations on
`_whisk_pgcstack`. Frame layout in linear memory:

    ┌────────────────────┐
    │ prev_frame : i64   │  offset 0
    │ n_slots    : i64   │  offset 8
    │ slot[0..n-1]: i64  │  offset 16, 24, 32, ...
    └────────────────────┘
"""
function lower_gc_frame_intrinsics!(mod::LLVM.Module)
    T_w   = LLVM.Int64Type()
    T_wp  = LLVM.PointerType(T_w)
    T_i8p = LLVM.PointerType(LLVM.Int8Type())

    pgcstack = LLVM.globals(mod)["_whisk_pgcstack"]
    whisk_alloc = LLVM.functions(mod)["whisk_gc_alloc"]
    whisk_alloc_ft = LLVM.function_type(whisk_alloc)

    # ── julia.new_gc_frame(n) -> ptr addrspace(_)
    if haskey(LLVM.functions(mod), "julia.new_gc_frame")
        intr = LLVM.functions(mod)["julia.new_gc_frame"]
        ret_ty = LLVM.return_type(LLVM.function_type(intr))
        for use in collect(LLVM.uses(intr))
            call = LLVM.user(use)::LLVM.CallInst
            n = LLVM.arguments(call)[1]
            n64 = if LLVM.value_type(n) == T_w
                n
            else
                @dispose b=LLVM.IRBuilder() begin
                    LLVM.position!(b, call)
                    LLVM.zext!(b, n, T_w)
                end
            end
            @dispose b=LLVM.IRBuilder() begin
                LLVM.position!(b, call)
                # bytes = 16 + 8*n  (header is 2 i64 fields)
                n_x8   = LLVM.shl!(b, n64, LLVM.ConstantInt(T_w, 3))
                bytes  = LLVM.add!(b, n_x8, LLVM.ConstantInt(T_w, 16))
                frame  = LLVM.call!(b, whisk_alloc_ft, whisk_alloc, [bytes])
                # prev_frame = 0 (set on push)
                prev_p = LLVM.inttoptr!(b, frame, T_wp)
                LLVM.store!(b, LLVM.ConstantInt(T_w, 0), prev_p)
                # n_slots = n
                n_addr = LLVM.add!(b, frame, LLVM.ConstantInt(T_w, 8))
                LLVM.store!(b, n64,
                              LLVM.inttoptr!(b, n_addr, T_wp))
                fr_ptr0 = LLVM.inttoptr!(b, frame, T_i8p)
                cast    = ret_ty isa LLVM.PointerType ?
                          LLVM.addrspacecast!(b, fr_ptr0, ret_ty) : frame
                LLVM.replace_uses!(call, cast)
            end
            LLVM.erase!(call)
        end
        isempty(LLVM.uses(intr)) && LLVM.erase!(intr)
    end

    # ── julia.push_gc_frame(frame, _n)
    if haskey(LLVM.functions(mod), "julia.push_gc_frame")
        intr = LLVM.functions(mod)["julia.push_gc_frame"]
        for use in collect(LLVM.uses(intr))
            call = LLVM.user(use)::LLVM.CallInst
            f_ptr = LLVM.arguments(call)[1]
            @dispose b=LLVM.IRBuilder() begin
                LLVM.position!(b, call)
                f_int = LLVM.ptrtoint!(b, f_ptr, T_w)
                top   = LLVM.load!(b, T_w, pgcstack)
                LLVM.store!(b, top, LLVM.inttoptr!(b, f_int, T_wp))
                LLVM.store!(b, f_int, pgcstack)
            end
            LLVM.erase!(call)
        end
        isempty(LLVM.uses(intr)) && LLVM.erase!(intr)
    end

    # ── julia.pop_gc_frame(frame)
    if haskey(LLVM.functions(mod), "julia.pop_gc_frame")
        intr = LLVM.functions(mod)["julia.pop_gc_frame"]
        for use in collect(LLVM.uses(intr))
            call = LLVM.user(use)::LLVM.CallInst
            f_ptr = LLVM.arguments(call)[1]
            @dispose b=LLVM.IRBuilder() begin
                LLVM.position!(b, call)
                f_int = LLVM.ptrtoint!(b, f_ptr, T_w)
                prev  = LLVM.load!(b, T_w, LLVM.inttoptr!(b, f_int, T_wp))
                LLVM.store!(b, prev, pgcstack)
            end
            LLVM.erase!(call)
        end
        isempty(LLVM.uses(intr)) && LLVM.erase!(intr)
    end

    # ── julia.get_pgcstack() -> ptr
    if haskey(LLVM.functions(mod), "julia.get_pgcstack")
        intr = LLVM.functions(mod)["julia.get_pgcstack"]
        ret_ty = LLVM.return_type(LLVM.function_type(intr))
        for use in collect(LLVM.uses(intr))
            call = LLVM.user(use)::LLVM.CallInst
            @dispose b=LLVM.IRBuilder() begin
                LLVM.position!(b, call)
                cast = ret_ty isa LLVM.PointerType ?
                       LLVM.addrspacecast!(b, pgcstack, ret_ty) :
                       LLVM.ptrtoint!(b, pgcstack, ret_ty)
                LLVM.replace_uses!(call, cast)
            end
            LLVM.erase!(call)
        end
        isempty(LLVM.uses(intr)) && LLVM.erase!(intr)
    end

    # ── julia.get_gc_frame_slot(frame, i) -> ptr
    if haskey(LLVM.functions(mod), "julia.get_gc_frame_slot")
        intr = LLVM.functions(mod)["julia.get_gc_frame_slot"]
        ret_ty = LLVM.return_type(LLVM.function_type(intr))
        for use in collect(LLVM.uses(intr))
            call = LLVM.user(use)::LLVM.CallInst
            f, i = LLVM.arguments(call)
            i64 = if LLVM.value_type(i) == T_w
                i
            else
                @dispose b=LLVM.IRBuilder() begin
                    LLVM.position!(b, call)
                    LLVM.zext!(b, i, T_w)
                end
            end
            @dispose b=LLVM.IRBuilder() begin
                LLVM.position!(b, call)
                f_int    = LLVM.ptrtoint!(b, f, T_w)
                i_x8     = LLVM.shl!(b, i64, LLVM.ConstantInt(T_w, 3))
                offset   = LLVM.add!(b, i_x8, LLVM.ConstantInt(T_w, 16))
                slot_int = LLVM.add!(b, f_int, offset)
                slot_p0  = LLVM.inttoptr!(b, slot_int, T_i8p)
                cast     = ret_ty isa LLVM.PointerType ?
                           LLVM.addrspacecast!(b, slot_p0, ret_ty) : slot_p0
                LLVM.replace_uses!(call, cast)
            end
            LLVM.erase!(call)
        end
        isempty(LLVM.uses(intr)) && LLVM.erase!(intr)
    end

    # ── julia.write_barrier — erase
    if haskey(LLVM.functions(mod), "julia.write_barrier")
        intr = LLVM.functions(mod)["julia.write_barrier"]
        for use in collect(LLVM.uses(intr))
            LLVM.erase!(LLVM.user(use)::LLVM.CallInst)
        end
        isempty(LLVM.uses(intr)) && LLVM.erase!(intr)
    end
end

"""
    rewrite_host_pointer_constants!(mod::LLVM.Module)

In Julia-emitted IR, `inttoptr (ConstantInt to ptr)` comes from
`pointer_from_objref` getting baked in by const-prop — always a host
pointer. wasm-internal addresses come back as register values from
`whisk_gc_alloc` calls, never as literal constants.

We try `Base.unsafe_pointer_to_objref` to identify each baked host
object, then build an equivalent wasm-side global so the IR's
follow-up loads see the right bytes:

  * `Memory{T}`           → a 16-byte struct global (length + data
                            ptr) with the data bytes inline
  * `String`              — same as `Memory{UInt8}` of the codeunits
  * unknown / unrecognised → `@_whisk_empty_memory` (16 zero bytes;
                              fine for objects that are passed as
                              ignored ccall args)

This is what makes `s = "hello"; codeunit(s, i)` actually return 'h'
when the optimizer const-folded the string.
"""
function rewrite_host_pointer_constants!(mod::LLVM.Module)
    empty_buf = _get_or_make_empty_buf!(mod)

    # ConstantExprs are owned by the LLVMContext, not the Module — the
    # SAME `inttoptr` constant can be referenced from instructions in
    # other modules that share our context (GPUCompiler runtime sub-jobs
    # whose modules have not yet been linked into ours). Calling
    # `replace_uses!(ce, ...)` would silently rewrite those foreign
    # instructions to reference our toplevel-module global, which the
    # verifier later rejects as a cross-module reference.
    #
    # Instead, walk every instruction in *this* module and rewrite each
    # operand slot in place via `LLVMSetOperand`. Foreign modules are
    # left untouched.
    replacements = IdDict{Int, Any}()  # host_addr → replacement global
    n = 0
    for fn in LLVM.functions(mod)
        LLVM.isdeclaration(fn) && continue
        for bb in LLVM.blocks(fn), inst in LLVM.instructions(bb)
            for (i, op) in enumerate(LLVM.operands(inst))
                op isa LLVM.ConstantExpr || continue
                LLVM.opcode(op) == LLVM.API.LLVMIntToPtr || continue
                src = LLVM.operands(op)[1]
                src isa LLVM.ConstantInt || continue
                host_addr = convert(Int, src)
                replacement = get!(replacements, host_addr) do
                    _reconstruct_host_object_in_wasm!(mod, host_addr, empty_buf)
                end
                LLVM.API.LLVMSetOperand(inst, i - 1, replacement)
                n += 1
            end
        end
    end

    # Julia's frontend also seeds module globals with host-pointer
    # initializers — these are the `@jl_global_NNNN = internal global
    # ptr inttoptr (i64 ADDR to ptr)` lines that appear after irgen for
    # type tags / `Type{T}` references / etc. Walk every global and
    # neuter such initializers: an inttoptr to a host VM address is
    # never valid in our wasm module, so substitute a null pointer.
    # The runtime never reads these for the current code paths (they
    # exist as defensive type-check material that the optimizer drops
    # when types are concrete), but leaving them as host pointers
    # produces invalid wasm relocations at link time.
    for gv in collect(LLVM.globals(mod))
        init = LLVM.initializer(gv)
        init === nothing && continue
        if init isa LLVM.ConstantExpr &&
           LLVM.opcode(init) == LLVM.API.LLVMIntToPtr
            ity = LLVM.value_type(init)
            null_val = LLVM.null(ity)
            LLVM.initializer!(gv, null_val)
            @assert LLVM.initializer(gv) === null_val ||
                    !(LLVM.initializer(gv) isa LLVM.ConstantExpr) "global init wasn't replaced for $(LLVM.name(gv))"
            n += 1
        end
    end

    # Same story for `external` declarations of `jl_*` host symbols
    # (`@jl_small_typeof`, `@jl_nothing`, `@jl_int64_type`, etc.). Make
    # them internal globals with a null/zero initializer of the
    # declared type so wasm-ld doesn't try to resolve a host symbol.
    for gv in collect(LLVM.globals(mod))
        LLVM.isdeclaration(gv) || continue
        nm = LLVM.name(gv)
        startswith(nm, "jl_") || continue
        ty = LLVM.global_value_type(gv)
        LLVM.initializer!(gv, LLVM.null(ty))
        LLVM.linkage!(gv, LLVM.API.LLVMInternalLinkage)
        n += 1
    end

    return n
end

function _get_or_make_empty_buf!(mod)
    name = "_whisk_empty_memory"
    if haskey(LLVM.globals(mod), name)
        return LLVM.globals(mod)[name]
    end
    ty = LLVM.ArrayType(LLVM.Int8Type(), 16)
    gv = LLVM.GlobalVariable(mod, ty, name)
    LLVM.initializer!(gv, LLVM.null(ty))
    LLVM.linkage!(gv, LLVM.API.LLVMInternalLinkage)
    return gv
end

# Try to read the host-VM object the address points at and snapshot
# it into a wasm global. Returns the global to replace `inttoptr addr`
# uses with. Falls back to the shared empty buffer when we can't
# identify or don't support the host object's type.
function _reconstruct_host_object_in_wasm!(mod::LLVM.Module, host_addr::Int, empty_buf)
    # Defensive: `unsafe_pointer_to_objref` is genuinely unsafe and can
    # SIGSEGV (try/catch doesn't help — segfaults bypass Julia's
    # exception machinery). Filter obvious non-objects first:
    #   * Address must be 16-byte-aligned (Julia heap alignment).
    #   * Address must be in a reasonable range (not a small int that
    #     happened to look like a pointer).
    # Then deref behind a try; treat any error or "not Memory/String"
    # as a safe fallback to the empty buffer.
    host_addr <= 0x100000      && return empty_buf
    host_addr % 16 != 0        && return empty_buf

    obj = try
        unsafe_pointer_to_objref(Ptr{Nothing}(host_addr))
    catch
        return empty_buf
    end
    if obj isa Memory{UInt8}
        return _make_wasm_memory_u8_global!(mod, obj, host_addr)
    elseif obj isa String
        return _make_wasm_string_global!(mod, codeunits(obj), host_addr)
    elseif obj isa Memory && isbitstype(eltype(obj))
        # Memory{T<:bitstype}: same shape as Memory{UInt8}; reuse the
        # byte-buffer machinery on the underlying bytes.
        T = eltype(obj)
        n = length(obj) * sizeof(T)
        bytes = Vector{UInt8}(undef, n)
        if n > 0
            GC.@preserve obj bytes begin
                ccall(:memcpy, Ptr{Cvoid},
                    (Ptr{Cvoid}, Ptr{Cvoid}, Csize_t),
                    pointer(bytes),
                    Base.unsafe_convert(Ptr{T}, obj),
                    Csize_t(n))
            end
        end
        return _make_wasm_memory_u8_global!(mod, bytes, host_addr)
    elseif obj isa Vector && isbitstype(eltype(obj))
        return _make_wasm_vector_bits_global!(mod, obj, host_addr)
    end
    return empty_buf
end

# Build (or reuse) a wasm global that materialises a `Memory{UInt8}`
# layout: 16-byte struct { i64 length, ptr data } followed by the
# actual data bytes. The struct's `data` field points at the inline
# bytes that immediately follow it.
const _WASM_MEMORY_GLOBAL_CACHE = IdDict{LLVM.Module, Dict{Int, Any}}()

function _make_wasm_vector_bits_global!(mod::LLVM.Module, vec, host_addr::Int)
    cache = get!(() -> Dict{Int, Any}(), _WASM_MEMORY_GLOBAL_CACHE, mod)
    haskey(cache, host_addr) && return cache[host_addr]

    T = eltype(vec)
    n = length(vec)
    elsz = Int(Base.aligned_sizeof(T))
    nbytes = n * elsz

    # 1) Raw data bytes (the Memory's inline payload).
    data_arr_ty = LLVM.ArrayType(LLVM.Int8Type(), nbytes)
    data_init   = if nbytes == 0
        LLVM.null(data_arr_ty)
    else
        raw = unsafe_wrap(Array, Ptr{UInt8}(Base.unsafe_convert(Ptr{T}, vec)),
                          nbytes; own = false)
        LLVM.ConstantArray(LLVM.Int8Type(),
            [LLVM.ConstantInt(LLVM.Int8Type(), Int(b)) for b in raw])
    end
    data_gv = LLVM.GlobalVariable(mod, data_arr_ty, "_whisk_vec_data_$(host_addr)")
    LLVM.initializer!(data_gv, data_init)
    LLVM.linkage!(data_gv, LLVM.API.LLVMInternalLinkage)
    LLVM.constant!(data_gv, true)

    # 2) Memory{T} struct (length + data ptr) so any path that loads
    # `vec.ref.mem` and dereferences finds a valid Memory.
    mem_struct_ty = LLVM.StructType([LLVM.Int64Type(), LLVM.PointerType(LLVM.Int8Type())])
    mem_init = LLVM.ConstantStruct(mem_struct_ty,
        LLVM.Constant[LLVM.ConstantInt(LLVM.Int64Type(), n), data_gv])
    mem_gv = LLVM.GlobalVariable(mod, mem_struct_ty, "_whisk_vec_mem_$(host_addr)")
    LLVM.initializer!(mem_gv, mem_init)
    LLVM.linkage!(mem_gv, LLVM.API.LLVMInternalLinkage)
    LLVM.constant!(mem_gv, true)

    # 3) Vector struct: {data_ptr, mem_ptr, length}. IR will:
    #     * load data_ptr at +0 for `arr[i]` reads
    #     * load length  at +16 for the bounds check
    #     * load mem_ptr at +8  if any path needs the Memory directly
    vec_struct_ty = LLVM.StructType([
        LLVM.PointerType(LLVM.Int8Type()),   # ptr_or_offset (data)
        LLVM.PointerType(LLVM.Int8Type()),   # mem
        LLVM.Int64Type(),                    # size
    ])
    vec_init = LLVM.ConstantStruct(vec_struct_ty,
        LLVM.Constant[data_gv, mem_gv, LLVM.ConstantInt(LLVM.Int64Type(), n)])
    vec_gv = LLVM.GlobalVariable(mod, vec_struct_ty, "_whisk_vec_$(host_addr)")
    LLVM.initializer!(vec_gv, vec_init)
    LLVM.linkage!(vec_gv, LLVM.API.LLVMInternalLinkage)
    LLVM.constant!(vec_gv, true)

    cache[host_addr] = vec_gv
    return vec_gv
end

function _make_wasm_memory_u8_global!(mod::LLVM.Module, bytes, host_addr::Int)
    cache = get!(() -> Dict{Int, Any}(), _WASM_MEMORY_GLOBAL_CACHE, mod)
    haskey(cache, host_addr) && return cache[host_addr]

    nbytes = length(bytes)
    # Layout: [u64 length][u64 placeholder for data ptr][bytes...]
    # (We can't put the real data ptr in an LLVM constant initializer
    #  — its value depends on the global's own address, which the
    #  linker resolves. Instead we keep the data inline immediately
    #  after the struct and emit a getelementptr at runtime to derive
    #  the data ptr. That requires Julia's IR to know the layout
    #  though, and it doesn't — it expects {length, data_ptr} where
    #  data_ptr is a real pointer to bytes.)
    #
    # Workaround: emit *two* separate globals, one for the data bytes
    # and one for the struct. The struct's `data` field is initialised
    # via an LLVM ConstantExpr that ptrtoint's the data global into i64.

    data_arr_ty = LLVM.ArrayType(LLVM.Int8Type(), nbytes)
    data_init   = if nbytes == 0
        LLVM.null(data_arr_ty)
    else
        LLVM.ConstantArray(LLVM.Int8Type(),
                           [LLVM.ConstantInt(LLVM.Int8Type(), Int(b)) for b in bytes])
    end
    data_gv = LLVM.GlobalVariable(mod, data_arr_ty,
                                   "_whisk_data_$(host_addr)")
    LLVM.initializer!(data_gv, data_init)
    LLVM.linkage!(data_gv, LLVM.API.LLVMInternalLinkage)
    LLVM.constant!(data_gv, true)

    # Struct: { i64 length, ptr data }. We need an LLVM struct type
    # matching what Julia's IR expects.
    struct_ty = LLVM.StructType([LLVM.Int64Type(), LLVM.PointerType(LLVM.Int8Type())])
    length_const = LLVM.ConstantInt(LLVM.Int64Type(), nbytes)
    data_ptr_const = data_gv  # pointer constant
    struct_init = LLVM.ConstantStruct(struct_ty, LLVM.Constant[length_const, data_ptr_const])

    struct_gv = LLVM.GlobalVariable(mod, struct_ty,
                                     "_whisk_mem_$(host_addr)")
    LLVM.initializer!(struct_gv, struct_init)
    LLVM.linkage!(struct_gv, LLVM.API.LLVMInternalLinkage)
    LLVM.constant!(struct_gv, true)

    cache[host_addr] = struct_gv
    return struct_gv
end

"""
Build a wasm global mirroring Julia's `String` runtime layout. Julia
accesses string bytes via `gep i8, host_addr, i64 8` (where `host_addr`
is the String object) then `+i-1`. We emit a flat byte array of
length `8 + length(bytes)`: the first 8 bytes are dummy header, the
remainder are the actual codeunits.
"""
function _make_wasm_string_global!(mod::LLVM.Module, bytes, host_addr::Int)
    cache = get!(() -> Dict{Int, Any}(), _WASM_MEMORY_GLOBAL_CACHE, mod)
    haskey(cache, host_addr) && return cache[host_addr]

    nbytes = length(bytes)
    total = 8 + nbytes
    ty = LLVM.ArrayType(LLVM.Int8Type(), total)

    consts = Vector{LLVM.Constant}(undef, total)
    for i in 1:8
        consts[i] = LLVM.ConstantInt(LLVM.Int8Type(), 0)
    end
    for (j, b) in enumerate(bytes)
        consts[8 + j] = LLVM.ConstantInt(LLVM.Int8Type(), Int(b))
    end
    init = LLVM.ConstantArray(LLVM.Int8Type(), consts)

    gv = LLVM.GlobalVariable(mod, ty, "_whisk_str_$(host_addr)")
    LLVM.initializer!(gv, init)
    LLVM.linkage!(gv, LLVM.API.LLVMInternalLinkage)
    LLVM.constant!(gv, true)

    cache[host_addr] = gv
    return gv
end

"""
    bypass_argument_error_checks!(mod::LLVM.Module)

Bypass Julia's mistruncated wasm bounds checks. Find every
conditional branch whose fail target ends with
`call jl_argument_error; unreachable` and rewrite to unconditionally
take the success target.
"""
function bypass_argument_error_checks!(mod::LLVM.Module)
    haskey(LLVM.functions(mod), "jl_argument_error") || return
    err_fn = LLVM.functions(mod)["jl_argument_error"]
    fail_blocks = LLVM.BasicBlock[]
    for use in collect(LLVM.uses(err_fn))
        call = LLVM.user(use)
        call isa LLVM.CallInst || continue
        push!(fail_blocks, LLVM.parent(call))
    end

    for fn in LLVM.functions(mod)
        LLVM.isdeclaration(fn) && continue
        for bb in LLVM.blocks(fn)
            term = LLVM.terminator(bb)
            term isa LLVM.BrInst || continue
            ops = LLVM.operands(term)
            length(ops) == 3 || continue
            true_tgt, false_tgt = ops[3], ops[2]
            true_is_fail  = true_tgt  in fail_blocks
            false_is_fail = false_tgt in fail_blocks
            true_is_fail == false_is_fail && continue
            keep = true_is_fail ? false_tgt : true_tgt
            @dispose b=LLVM.IRBuilder() begin
                LLVM.position!(b, term)
                LLVM.br!(b, keep)
            end
            LLVM.erase!(term)
        end
    end
end
