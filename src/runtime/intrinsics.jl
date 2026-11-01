## compiler-rt helpers for wasm
##
## On targets without native i128, LLVM's late codegen emits libcalls to a
## handful of compiler-rt symbols when it sees i128 ops. Julia's `sin` /
## `cos` / `tan` for `Float64` use Int128 inside Payne-Hanek argument
## reduction, so even basic trig pulls these in.
##
## We provide them as plain Julia functions using only UInt64 / UInt32
## arithmetic. GPUCompiler compiles them through the same pipeline as user
## code; their LLVM symbols are renamed at link-in time to the C ABI names
## (`__multi3`, `__lshrti3`, `__ashlti3`) that LLVM's lowering expects.
##
## Wasm calling convention (sret pointer + decomposed i128 as 2× i64):
##   void __multi3 (i128* result, i64 a_lo, i64 a_hi, i64 b_lo, i64 b_hi)
##   void __lshrti3(i128* result, i64 a_lo, i64 a_hi, i32 shift)
##   void __ashlti3(i128* result, i64 a_lo, i64 a_hi, i32 shift)

# Upper 64 bits of the full 128-bit product `a * b` using only UInt64 ops.
# Decomposes each 64-bit operand into two 32-bit halves and recombines —
# all four partial products fit in UInt64 because each operand is < 2^32.
@inline function whisk_umulh64(a::UInt64, b::UInt64)::UInt64
    aL = a & 0xffffffff
    aH = a >> 32
    bL = b & 0xffffffff
    bH = b >> 32

    pp_ll = aL * bL
    pp_hl = aH * bL
    pp_lh = aL * bH
    pp_hh = aH * bH

    mid = (pp_ll >> 32) + (pp_hl & 0xffffffff) + (pp_lh & 0xffffffff)
    return pp_hh + (pp_hl >> 32) + (pp_lh >> 32) + (mid >> 32)
end

# (a * b) mod 2^128 — only the low 128 bits matter (LLVM handles widening
# elsewhere). Splits the work into four UInt64 multiplies.
function whisk_multi3(result::Ptr{UInt64},
                      a_lo::UInt64, a_hi::UInt64,
                      b_lo::UInt64, b_hi::UInt64)::Nothing
    low  = a_lo * b_lo
    high = whisk_umulh64(a_lo, b_lo) + a_hi * b_lo + a_lo * b_hi
    unsafe_store!(result, low,  1)
    unsafe_store!(result, high, 2)
    return nothing
end

# Logical shift right of a 128-bit value by `b` bits.
function whisk_lshrti3(result::Ptr{UInt64},
                       a_lo::UInt64, a_hi::UInt64, b::UInt32)::Nothing
    bb = UInt64(b) & UInt64(127)
    new_lo, new_hi = if bb == 0
        (a_lo, a_hi)
    elseif bb < 64
        ((a_lo >> bb) | (a_hi << (UInt64(64) - bb)), a_hi >> bb)
    else
        (a_hi >> (bb - UInt64(64)), UInt64(0))
    end
    unsafe_store!(result, new_lo, 1)
    unsafe_store!(result, new_hi, 2)
    return nothing
end

# Arithmetic / logical shift left of a 128-bit value by `b` bits (left
# shift is the same for signed and unsigned).
function whisk_ashlti3(result::Ptr{UInt64},
                       a_lo::UInt64, a_hi::UInt64, b::UInt32)::Nothing
    bb = UInt64(b) & UInt64(127)
    new_lo, new_hi = if bb == 0
        (a_lo, a_hi)
    elseif bb < 64
        (a_lo << bb, (a_hi << bb) | (a_lo >> (UInt64(64) - bb)))
    else
        (UInt64(0), a_lo << (bb - UInt64(64)))
    end
    unsafe_store!(result, new_lo, 1)
    unsafe_store!(result, new_hi, 2)
    return nothing
end

## ─────────────────────────────────────────────────────────────────────
## Boxing helpers (`ijl_box_*`)
## ─────────────────────────────────────────────────────────────────────
##
## Julia emits `call ptr @ijl_box_int64(i64 v)` etc. when it needs to
## materialise an `Any`-tagged heap value — typically inside exception
## constructors and dynamic-dispatch call frames. The C ABI is:
##
##   `jl_value_t* jl_box_int64 (int64_t v)`     etc.
##
## The returned pointer points at the value; the type tag lives at
## offset −8 (Julia's standard heap-object layout). Our wasm runtime
## doesn't use the host's GC tracking — boxes live on our own linear-
## memory heap (allocated by `whisk_gc_alloc`). The type tag is the
## linked address of the wasm-side DataType descriptor global
## (`whisk_typetag_<T>` extern, defined by `define_box_typetags!`), so
## `typeof` / `isa` / typeassert tag compares work on box values.

# `ccall("extern <sym>", llvmcall, …)` is the GPUCompiler-idiomatic way
# to emit a direct `call @whisk_gc_alloc(i64 …)` that resolves at link
# time. The `Base.llvmcall((decl, body), …)` tuple form is NOT safe
# here: under the GPUCompiler interpreter the tuple argument doesn't
# const-fold into the intrinsic, and Julia's codegen then silently
# emits `jl_error("llvmcall requires a constant")` — which Whisk's
# throw lowering turns into an unconditional trap, replacing the
# allocator body with `unreachable` (bit us via every `ijl_box_*`).
@inline function _whisk_gc_alloc(sz::UInt64)::UInt64
    ccall("extern whisk_gc_alloc", llvmcall, UInt64, (UInt64,), sz)
end

# Allocate a Julia-style box. With unified type descriptors the GC
# header's layout word at `payload[-8]` IS the Julia type tag
# (`whisk_gc_alloc_typed` stores it there), so no extra tag word is
# needed — and boxes whose payload contains heap references are
# scanned precisely per the descriptor's layout prefix.
@inline function _whisk_alloc_box(payload_size::UInt64, type_tag::UInt64)::UInt64
    ccall("extern whisk_gc_alloc_typed", llvmcall, UInt64, (UInt64, UInt64),
          payload_size, type_tag)
end

# Per-type tag accessors. Each emits `call i64 @whisk_typetag_<T>()` —
# an extern the compile pipeline defines after linking (see
# `define_box_typetags!` in compiler/host_globals.jl): it returns the
# linked address of the wasm-side DataType descriptor global for `T`,
# the SAME global that `rewrite_host_pointer_constants!` substitutes
# for baked host `DataType` pointers at `isa`/typeassert compare sites.
# Boxes tagged this way therefore pass runtime type checks.
@inline _whisk_typetag_Int64()   = ccall("extern whisk_typetag_Int64",   llvmcall, UInt64, ())
@inline _whisk_typetag_Int32()   = ccall("extern whisk_typetag_Int32",   llvmcall, UInt64, ())
@inline _whisk_typetag_Int16()   = ccall("extern whisk_typetag_Int16",   llvmcall, UInt64, ())
@inline _whisk_typetag_Int8()    = ccall("extern whisk_typetag_Int8",    llvmcall, UInt64, ())
@inline _whisk_typetag_UInt8()   = ccall("extern whisk_typetag_UInt8",   llvmcall, UInt64, ())
@inline _whisk_typetag_UInt16()  = ccall("extern whisk_typetag_UInt16",  llvmcall, UInt64, ())
@inline _whisk_typetag_UInt32()  = ccall("extern whisk_typetag_UInt32",  llvmcall, UInt64, ())
@inline _whisk_typetag_UInt64()  = ccall("extern whisk_typetag_UInt64",  llvmcall, UInt64, ())
@inline _whisk_typetag_Float64() = ccall("extern whisk_typetag_Float64", llvmcall, UInt64, ())
@inline _whisk_typetag_Float32() = ccall("extern whisk_typetag_Float32", llvmcall, UInt64, ())
@inline _whisk_typetag_Char()    = ccall("extern whisk_typetag_Char",    llvmcall, UInt64, ())
@inline _whisk_typetag_Bool()    = ccall("extern whisk_typetag_Bool",    llvmcall, UInt64, ())

function whisk_jl_box_int64(v::Int64)::Ptr{Cvoid}
    addr = _whisk_alloc_box(UInt64(8), _whisk_typetag_Int64())
    unsafe_store!(Ptr{Int64}(addr), v, 1)
    return Ptr{Cvoid}(addr)
end

function whisk_jl_box_uint64(v::UInt64)::Ptr{Cvoid}
    addr = _whisk_alloc_box(UInt64(8), _whisk_typetag_UInt64())
    unsafe_store!(Ptr{UInt64}(addr), v, 1)
    return Ptr{Cvoid}(addr)
end

function whisk_jl_box_int32(v::Int32)::Ptr{Cvoid}
    addr = _whisk_alloc_box(UInt64(8), _whisk_typetag_Int32())
    unsafe_store!(Ptr{Int32}(addr), v, 1)
    return Ptr{Cvoid}(addr)
end

function whisk_jl_box_uint8(v::UInt8)::Ptr{Cvoid}
    addr = _whisk_alloc_box(UInt64(8), _whisk_typetag_UInt8())
    unsafe_store!(Ptr{UInt8}(addr), v, 1)
    return Ptr{Cvoid}(addr)
end

function whisk_jl_box_float64(v::Float64)::Ptr{Cvoid}
    addr = _whisk_alloc_box(UInt64(8), _whisk_typetag_Float64())
    unsafe_store!(Ptr{Float64}(addr), v, 1)
    return Ptr{Cvoid}(addr)
end

function whisk_jl_box_float32(v::Float32)::Ptr{Cvoid}
    addr = _whisk_alloc_box(UInt64(8), _whisk_typetag_Float32())
    unsafe_store!(Ptr{Float32}(addr), v, 1)
    return Ptr{Cvoid}(addr)
end

function whisk_jl_box_int8(v::Int8)::Ptr{Cvoid}
    addr = _whisk_alloc_box(UInt64(8), _whisk_typetag_Int8())
    unsafe_store!(Ptr{Int8}(addr), v, 1)
    return Ptr{Cvoid}(addr)
end

function whisk_jl_box_int16(v::Int16)::Ptr{Cvoid}
    addr = _whisk_alloc_box(UInt64(8), _whisk_typetag_Int16())
    unsafe_store!(Ptr{Int16}(addr), v, 1)
    return Ptr{Cvoid}(addr)
end

function whisk_jl_box_uint16(v::UInt16)::Ptr{Cvoid}
    addr = _whisk_alloc_box(UInt64(8), _whisk_typetag_UInt16())
    unsafe_store!(Ptr{UInt16}(addr), v, 1)
    return Ptr{Cvoid}(addr)
end

function whisk_jl_box_uint32(v::UInt32)::Ptr{Cvoid}
    addr = _whisk_alloc_box(UInt64(8), _whisk_typetag_UInt32())
    unsafe_store!(Ptr{UInt32}(addr), v, 1)
    return Ptr{Cvoid}(addr)
end

# `jl_box_char` takes the char's UTF-32 codepoint payload as uint32.
function whisk_jl_box_char(v::UInt32)::Ptr{Cvoid}
    addr = _whisk_alloc_box(UInt64(8), _whisk_typetag_Char())
    unsafe_store!(Ptr{UInt32}(addr), v, 1)
    return Ptr{Cvoid}(addr)
end

function whisk_jl_box_bool(v::Int8)::Ptr{Cvoid}
    addr = _whisk_alloc_box(UInt64(8), _whisk_typetag_Bool())
    unsafe_store!(Ptr{Int8}(addr), v, 1)
    return Ptr{Cvoid}(addr)
end

## ─────────────────────────────────────────────────────────────────────
## GC growth helper (`whisk_ensure_capacity`)
## ─────────────────────────────────────────────────────────────────────
##
## Called by `whisk_gc_alloc`'s bump path before storing the header at
## `end_addr`. If `end_addr` is past the end of linear memory, run
## mark+sweep first (the GC actually doing its job) — that frees dead
## allocations into the free list for any *subsequent* alloc. Then, if
## the live set still doesn't fit, call `memory.grow`.
##
## We can't write the whole allocator in Julia (cross-module access to
## the internal-linkage `_whisk_*` globals via `Base.llvmcall(... external
## global ...)` doesn't survive GPUCompiler's pipeline — sub-compiles
## stub the function body), so the bump/free-list IR stays in
## `runtime/gc.jl`. This helper handles only the parts that don't need
## global access.

# wasm `memory.size` / `memory.grow` intrinsics via the
# `ccall(<name>, llvmcall, …)` form — emits a direct
# `call i64 @llvm.wasm.memory.<op>.i64(...)` in the function body that
# wasm-ld lowers to the corresponding wasm opcode. (Tried
# `Base.llvmcall(("declare …", body), …)` first — the host frontend
# rejected those LLVM-intrinsic declarations with `Module IR does not
# contain …`; no `extern` prefix on the ccall name either, that form
# is for `Base.llvmcall`-emitted externs, not LLVM intrinsics.)
# `@inline` so they fold into `whisk_ensure_capacity` instead of going
# through Julia's methodtable (which on the wasm sub-job would route
# the calls through `@jl_global_*` pointers that don't survive).
@inline whisk_memory_size_pages() =
    ccall("llvm.wasm.memory.size.i64", llvmcall, UInt64, (Int32,), Int32(0))

@inline whisk_memory_grow_pages(delta::UInt64) =
    ccall("llvm.wasm.memory.grow.i64", llvmcall, UInt64, (Int32, UInt64),
          Int32(0), delta)

# When the bump pointer would land past the current memory size,
# `memory.grow` enough pages to fit. We DON'T collect inside this
# helper: collecting mid-allocation would mean mark phase runs while
# the wasm function is in the middle of its work, with locals in
# wasm SSA registers that mark can't see — those locals would be
# swept and reuse would corrupt them.
#
# Instead, collect happens at well-defined safepoints between
# wasm-call invocations, when the pgcstack is empty and JS holds
# the only roots (its pinned input/output buffers). The host calls
# `instance.exports.whisk_gc_collect()` to do this — typically
# right after reading a `route_full` result, or on idle in a
# long-running app.
function whisk_ensure_capacity(end_addr::UInt64)::Nothing
    PAGE = UInt64(65536)
    cur_bytes = whisk_memory_size_pages() * PAGE
    if end_addr > cur_bytes
        delta_bytes = end_addr - cur_bytes
        delta_pages = (delta_bytes + PAGE - UInt64(1)) ÷ PAGE
        whisk_memory_grow_pages(delta_pages)
    end
    return nothing
end

## ─────────────────────────────────────────────────────────────────────
## Runtime field access (`ijl_get_nth_field_checked`)
## ─────────────────────────────────────────────────────────────────────
##
## Julia emits `call ptr @ijl_get_nth_field_checked(ptr v, i64 i)` when
## it needs `getfield(v, i)` on a value whose type it only knows at
## runtime (e.g. iterating a heterogeneous vararg tuple in
## `print_to_string`). We walk the unified type descriptor that
## `obj[-8] & ~15` points at (see runtime/layout.jl for the wire
## format): pointer fields load the slot; inline fields are re-boxed
## with the field's own descriptor as the tag.

# Runtime-helper error channel. `ccall("extern js_console", llvmcall,…)`
# rather than `ccall(:js_console, …)` — in sub-jobs (toplevel=false) the
# bare-symbol form lowers to an `ijl_load_and_lookup` PLT thunk that
# wasm can't run; the extern form emits a direct call against the
# js_console import. The `llvm.trap` intrinsic ccall is the
# sub-job-safe halt (same pattern as `whisk_memory_size_pages`).
@noinline function _whisk_rt_error(msg::String)::Nothing
    GC.@preserve msg ccall("extern js_console", llvmcall, Cvoid,
                           (Int32, Ptr{UInt8}, UInt64),
                           Int32(2), pointer(msg), UInt64(sizeof(msg)))
    ccall("llvm.trap", llvmcall, Cvoid, ())
    return nothing
end

# `jl_f_tuple` lowering target: plain Julia, sub-compiled per arity at
# the call site (`lower_julia_call_builtins!`). The tuple construction,
# boxing and type tagging all go through the normal compile pipeline —
# no hand-emitted IR. `inferencebarrier` forces the `Any` (boxed-ptr)
# return ABI the `julia.call` site expects.
@noinline whisk_rt_tuple(args...) = Base.inferencebarrier(args)

function whisk_jl_get_nth_field_checked(v::Ptr{Cvoid}, i::UInt64)::Ptr{Cvoid}
    a = UInt64(reinterpret(UInt, v))
    desc = unsafe_load(Ptr{UInt64}(a - UInt64(8))) & ~UInt64(15)
    if desc == 0
        _whisk_rt_error("getfield: object carries no type descriptor (layout=0 allocation)")
        return Ptr{Cvoid}(0)
    end
    nptrs = unsafe_load(Ptr{UInt64}(desc + UInt64(8)))
    # Header: {elsz, nptrs, objectid, offsets[nptrs]} — meta follows.
    meta = desc + UInt64(24) + (nptrs << 3)
    nf = unsafe_load(Ptr{UInt64}(meta + UInt64(8)))
    if i >= nf
        _whisk_rt_error("getfield: field index out of bounds")
        return Ptr{Cvoid}(0)
    end
    fields = desc + unsafe_load(Ptr{UInt64}(meta + UInt64(16)))
    rec   = fields + i * UInt64(24)
    off   = unsafe_load(Ptr{UInt64}(rec))
    fdesc = unsafe_load(Ptr{UInt64}(rec + UInt64(8)))
    kind  = unsafe_load(Ptr{UInt64}(rec + UInt64(16)))
    if kind == UInt64(0)
        return Ptr{Cvoid}(unsafe_load(Ptr{UInt64}(a + off)))
    elseif kind == UInt64(1)
        fnptrs = unsafe_load(Ptr{UInt64}(fdesc + UInt64(8)))
        fmeta  = fdesc + UInt64(24) + (fnptrs << 3)
        sz     = unsafe_load(Ptr{UInt64}(fmeta))
        box = _whisk_alloc_box(sz, fdesc)
        ccall(:memcpy, Ptr{Cvoid}, (Ptr{Cvoid}, Ptr{Cvoid}, Csize_t),
              Ptr{Cvoid}(box), Ptr{Cvoid}(a + off), Csize_t(sz))
        return Ptr{Cvoid}(box)
    end
    _whisk_rt_error("getfield: unsupported field kind (inline union field?)")
    return Ptr{Cvoid}(0)
end

## `jl_egal__unboxed(a, b, dt)` — egal for unboxed/inline values of the
## same type: bitwise payload compare over the descriptor's
## payload_size. Pointer fields compare by identity, matching egal for
## mutable references. DEBT(#egal-strings): immutable ref fields whose
## CONTENTS are equal but identities differ (two equal heap Strings)
## compare unequal here; host egal recurses. Revisit with the data
## image (interning makes most such literals identical anyway).
function whisk_jl_egal_unboxed(a::Ptr{Cvoid}, b::Ptr{Cvoid}, dt::Ptr{Cvoid})::Int32
    d = UInt64(reinterpret(UInt, dt))
    d == 0 && return Int32(a == b)
    nptrs = unsafe_load(Ptr{UInt64}(d + UInt64(8)))
    sz = unsafe_load(Ptr{UInt64}(d + UInt64(24) + (nptrs << 3)))
    pa = Ptr{UInt8}(a); pb = Ptr{UInt8}(b)
    i = UInt64(0)
    while i < sz
        unsafe_load(pa, Int(i) + 1) == unsafe_load(pb, Int(i) + 1) ||
            return Int32(0)
        i += UInt64(1)
    end
    return Int32(1)
end

# Registry: C ABI symbol name → (Julia function, signature).
# The compile pipeline iterates this, sub-compiles each entry through
# GPUCompiler, renames the entry symbol, and links into the user's module.
const COMPILER_RT_HELPERS = (
    ("__multi3",        whisk_multi3,         Tuple{Ptr{UInt64}, UInt64, UInt64, UInt64, UInt64}),
    ("__lshrti3",       whisk_lshrti3,        Tuple{Ptr{UInt64}, UInt64, UInt64, UInt32}),
    ("__ashlti3",       whisk_ashlti3,        Tuple{Ptr{UInt64}, UInt64, UInt64, UInt32}),
    ("ijl_box_int64",   whisk_jl_box_int64,   Tuple{Int64}),
    ("ijl_box_int32",   whisk_jl_box_int32,   Tuple{Int32}),
    ("ijl_box_int16",   whisk_jl_box_int16,   Tuple{Int16}),
    ("ijl_box_int8",    whisk_jl_box_int8,    Tuple{Int8}),
    ("ijl_box_uint8",   whisk_jl_box_uint8,   Tuple{UInt8}),
    ("ijl_box_uint16",  whisk_jl_box_uint16,  Tuple{UInt16}),
    ("ijl_box_uint32",  whisk_jl_box_uint32,  Tuple{UInt32}),
    ("ijl_box_uint64",  whisk_jl_box_uint64,  Tuple{UInt64}),
    ("ijl_box_float64", whisk_jl_box_float64, Tuple{Float64}),
    ("ijl_box_float32", whisk_jl_box_float32, Tuple{Float32}),
    ("ijl_box_char",    whisk_jl_box_char,    Tuple{UInt32}),
    ("ijl_box_bool",    whisk_jl_box_bool,    Tuple{Int8}),
    ("whisk_ensure_capacity", whisk_ensure_capacity, Tuple{UInt64}),
    ("ijl_get_nth_field_checked", whisk_jl_get_nth_field_checked, Tuple{Ptr{Cvoid}, UInt64}),
    ("jl_egal__unboxed",  whisk_jl_egal_unboxed, Tuple{Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}}),
    ("ijl_egal__unboxed", whisk_jl_egal_unboxed, Tuple{Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}}),
)
