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
## memory heap (allocated by `whisk_gc_alloc`). The type tag is
## currently a zero placeholder; it will become a real type id when we
## need to differentiate boxes for dynamic dispatch (Step 4 of the
## next-phase plan).

# Direct LLVM call to `whisk_gc_alloc`. Goes via `llvmcall` rather than
# `ccall` because these helpers sub-compile with `toplevel = false` and
# Julia would otherwise route a bare-symbol ccall through
# `ijl_load_and_lookup` (the runtime symbol-resolution path), which
# can't run in our wasm module. `llvmcall` emits a direct
# `call @whisk_gc_alloc(i64 ...)` that resolves at link time against
# the parent module's allocator definition.
@inline function _whisk_gc_alloc(sz::UInt64)::UInt64
    Base.llvmcall(
        ("declare i64 @whisk_gc_alloc(i64)",
         """
         %r = call i64 @whisk_gc_alloc(i64 %0)
         ret i64 %r
         """),
        UInt64, Tuple{UInt64}, sz)
end

# Allocate a Julia-style box: 8-byte type tag header + payload bytes.
# Returns the address of the payload (so caller sees `box[-8]` = tag,
# `box[0..]` = value), matching Julia's heap layout.
@inline function _whisk_alloc_box(payload_size::UInt64, type_tag::UInt64)::UInt64
    base = _whisk_gc_alloc(payload_size + UInt64(8))
    unsafe_store!(Ptr{UInt64}(base), type_tag, 1)
    return base + UInt64(8)
end

function whisk_jl_box_int64(v::Int64)::Ptr{Cvoid}
    addr = _whisk_alloc_box(UInt64(8), UInt64(0))
    unsafe_store!(Ptr{Int64}(addr), v, 1)
    return Ptr{Cvoid}(addr)
end

function whisk_jl_box_uint64(v::UInt64)::Ptr{Cvoid}
    addr = _whisk_alloc_box(UInt64(8), UInt64(0))
    unsafe_store!(Ptr{UInt64}(addr), v, 1)
    return Ptr{Cvoid}(addr)
end

function whisk_jl_box_int32(v::Int32)::Ptr{Cvoid}
    addr = _whisk_alloc_box(UInt64(8), UInt64(0))
    unsafe_store!(Ptr{Int32}(addr), v, 1)
    return Ptr{Cvoid}(addr)
end

function whisk_jl_box_uint8(v::UInt8)::Ptr{Cvoid}
    addr = _whisk_alloc_box(UInt64(8), UInt64(0))
    unsafe_store!(Ptr{UInt8}(addr), v, 1)
    return Ptr{Cvoid}(addr)
end

function whisk_jl_box_float64(v::Float64)::Ptr{Cvoid}
    addr = _whisk_alloc_box(UInt64(8), UInt64(0))
    unsafe_store!(Ptr{Float64}(addr), v, 1)
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

# Registry: C ABI symbol name → (Julia function, signature).
# The compile pipeline iterates this, sub-compiles each entry through
# GPUCompiler, renames the entry symbol, and links into the user's module.
const COMPILER_RT_HELPERS = (
    ("__multi3",        whisk_multi3,         Tuple{Ptr{UInt64}, UInt64, UInt64, UInt64, UInt64}),
    ("__lshrti3",       whisk_lshrti3,        Tuple{Ptr{UInt64}, UInt64, UInt64, UInt32}),
    ("__ashlti3",       whisk_ashlti3,        Tuple{Ptr{UInt64}, UInt64, UInt64, UInt32}),
    ("ijl_box_int64",   whisk_jl_box_int64,   Tuple{Int64}),
    ("ijl_box_int32",   whisk_jl_box_int32,   Tuple{Int32}),
    ("ijl_box_uint8",   whisk_jl_box_uint8,   Tuple{UInt8}),
    ("ijl_box_uint64",  whisk_jl_box_uint64,  Tuple{UInt64}),
    ("ijl_box_float64", whisk_jl_box_float64, Tuple{Float64}),
    ("whisk_ensure_capacity", whisk_ensure_capacity, Tuple{UInt64}),
)
