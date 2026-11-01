## Precise-mark layout descriptors.
##
## The original GC was conservative: every aligned 8-byte word in a
## live object's payload was treated as a candidate heap pointer and
## the mark walker followed it. With ~71 MB of pinned input data and
## a multi-GB heap the false-positive rate was high enough that calls
## resurrected the previous call's ~200 MB graph wholesale, monotonic
## heap growth, multi-second `whisk_gc_collect`.
##
## Now: at compile time we emit a per-type **layout descriptor** for
## every Julia DataType that ever reaches `julia.gc_alloc_obj` or
## `jl_alloc_genericmemory_unchecked`. The descriptor encodes exactly
## which byte offsets contain heap pointers — nothing else gets
## scanned. Mark walks layout, follows real pointers, never imagines
## one out of payload bytes. False-positive rate: zero.
##
## Wire format (one wasm-side rodata global per type, 8-byte aligned):
##
##     LayoutDescriptor:
##       u64 elsz                  // 0 = fixed-size object,
##                                 // !=0 = Memory<T> with this elem size
##       u64 nptrs                 // count of pointer offsets
##       u64 offsets[nptrs]        // byte offsets, applied within the
##                                 // object (fixed) or within each
##                                 // T element (Memory)
##
## Layout-descriptor pointers are stored in the GC header at offset 8,
## with the mark/free flag bits packed into the low 3 bits (descriptors
## are always 8-byte aligned by construction).
##
## A `layout_ptr == 0` value means "raw bytes, no pointers" — used for
## explicit `Vector{UInt8}` storage, JS-allocated buffers, and
## bitstype Memory data where the per-element pointer count is 0.

# ── Layout computation (host-side, at compile time) ──────────────────

"""
    pointer_offsets(T::DataType) → Vector{UInt64}

Compute the byte offsets within an inline-stored instance of T at
which heap pointers live. Walks fields recursively for nested
immutable struct fields. Returns empty for `isbitstype(T)`.

Used as the per-element offset list for a `Memory{T}` descriptor and
as the within-payload offset list for a fixed-size object descriptor.
"""
function pointer_offsets(@nospecialize(T::DataType))::Vector{UInt64}
    offsets = UInt64[]
    pointer_offsets!(offsets, T, UInt64(0))
    return offsets
end

function pointer_offsets!(out::Vector{UInt64},
                          @nospecialize(T::DataType),
                          base::UInt64)
    isbitstype(T) && return  # primitive bytes, no refs
    if !isconcretetype(T) || isabstracttype(T)
        # Abstract / Union types: anything could be there. Treat the
        # whole thing as a single pointer slot at `base`. Caller can
        # decide whether that's appropriate (it usually isn't — the
        # alloc site for an abstract type is a sign that the wasm
        # target has hit dynamic dispatch we don't support yet).
        push!(out, base)
        return
    end
    nf = fieldcount(T)
    for i in 1:nf
        ft = fieldtype(T, i)
        off = base + UInt64(fieldoffset(T, i))
        if ft isa DataType && isbitstype(ft)
            continue
        elseif ft isa DataType && Base.allocatedinline(ft)
            # Inline-stored immutable with refs — recurse. Julia's own
            # storage predicate; inline recursion cannot cycle (the
            # type would have infinite size). The previous hand-rolled
            # rule misclassified some types and recursed forever.
            pointer_offsets!(out, ft, off)
        else
            # Pointer-stored: mutable, abstract, union, ref types.
            push!(out, off)
        end
    end
    return
end

"""
    memory_element_layout(::Type{Memory{T}}) → (elsz, ptr_offsets)

For a `Memory{T}`, return the per-element byte size and the
within-element pointer offsets. Three cases:

  * `isbitstype(T)` — `(sizeof(T), [])`. No pointers; mark walker
    can skip the scan loop entirely.
  * boxed elements (mutable T, `Array`, `String`, etc.) —
    `(8, [0])`. Each slot is a single 8-byte heap pointer.
  * inline-stored immutable struct with refs (e.g. `Tile`) —
    `(sizeof(T), pointer_offsets(T))`. Each slot is `sizeof(T)`
    bytes; pointers live at the same offsets within every element.
"""
function memory_element_layout(@nospecialize(memT::DataType))
    @assert memT <: Memory "expected Memory{T}, got $memT"
    # Memory{T}'s type parameters are (atomic_kind::Symbol, T, addrspace);
    # element type lives at position 2 in Julia 1.11+.
    T = memT.parameters[2]   # may be a UnionAll/abstract (Memory{Type}, …)
    if T isa DataType && isbitstype(T)
        return (UInt64(Base.aligned_sizeof(T)), UInt64[])
    end
    # Box-vs-inline per Julia's own storage predicate.
    if !(T isa DataType) || !Base.allocatedinline(T)
        return (UInt64(8), UInt64[0])
    end
    # Inline-stored immutable struct with pointer fields.
    return (UInt64(Base.aligned_sizeof(T)), pointer_offsets(T))
end

# ── Wasm-side descriptor emission ────────────────────────────────────

# A LayoutDescriptor in linear memory is just a sequence of i64 words:
#     [elsz, nptrs, offsets...]
# Stored at 8-byte alignment so the low 3 bits of any descriptor
# pointer are guaranteed zero (we use those for mark/free flags in
# the GC header).

"Module-level cache: one descriptor global per (elsz, ptr_offsets) tuple."
const _DESCRIPTOR_CACHE_KEY = "_whisk_layout_cache"

function _descriptor_cache(mod::LLVM.Module)
    md = LLVM.metadata(mod)
    # Use a per-module Dict keyed on (elsz, offsets) for memoisation
    # within one compile. Stored as a Julia object via `mod`'s
    # operand bundle is awkward; we just use a side dict keyed by
    # `objectid(mod)` for simplicity. Valid for one compile_to_wasm
    # invocation since we hold the module object the whole time.
    get!(_descriptor_cache_dict, objectid(mod)) do
        # Values are global NAMES, not GlobalVariable handles —
        # GPUCompiler's addrspace-removal pass clones every global
        # (handles dangle across pipeline phases; names survive).
        Dict{Tuple{UInt64, Vector{UInt64}}, String}()
    end
end
const _descriptor_cache_dict = Dict{UInt64, Dict{Tuple{UInt64, Vector{UInt64}}, String}}()

"""
    get_or_emit_layout_descriptor!(mod, elsz, ptr_offsets) → LLVM.GlobalVariable

Emit (or reuse) a LayoutDescriptor wasm-side global for the given
element size + pointer-offset list. Returns the global; its address
is the descriptor pointer that goes into the GC header.

`elsz == 0` means a fixed-size object (offsets within payload);
`elsz != 0` means a `Memory<T>` (offsets within each T element).
"""
function get_or_emit_layout_descriptor!(mod::LLVM.Module,
                                          elsz::UInt64,
                                          ptr_offsets::Vector{UInt64})
    cache = _descriptor_cache(mod)
    key = (elsz, copy(ptr_offsets))
    if haskey(cache, key) && haskey(LLVM.globals(mod), cache[key])
        return LLVM.globals(mod)[cache[key]]
    end

    T_w = LLVM.Int64Type()
    n   = length(ptr_offsets)
    arr_ty = LLVM.ArrayType(T_w, 3 + n)

    # Constant array literal: [elsz, nptrs, objectid=0, off0, off1, …].
    # The objectid slot at +16 is part of the descriptor header (tag
    # switches load it); shape-keyed anonymous layouts carry 0.
    elements = LLVM.Constant[
        LLVM.ConstantInt(T_w, elsz),
        LLVM.ConstantInt(T_w, n),
        LLVM.ConstantInt(T_w, 0),
    ]
    for off in ptr_offsets
        push!(elements, LLVM.ConstantInt(T_w, off))
    end
    init = LLVM.ConstantArray(T_w, elements)

    # Unique name per (module, descriptor) — count globals manually
    # since LLVM.ModuleGlobalSet doesn't implement length.
    n_existing = 0
    for _ in LLVM.globals(mod)
        n_existing += 1
    end
    name = string("_whisk_layout_", n_existing)
    gv = LLVM.GlobalVariable(mod, arr_ty, name)
    LLVM.initializer!(gv, init)
    LLVM.linkage!(gv, LLVM.API.LLVMInternalLinkage)
    LLVM.constant!(gv, true)
    LLVM.alignment!(gv, 8)

    # Pin via @llvm.used so optimizer DCE doesn't drop a descriptor
    # whose only references will be inserted by post-optimize passes.
    LLVM.API.LLVMAppendToUsed(mod, [gv], 1)

    cache[key] = name
    return gv
end

## ── Unified per-type descriptors ─────────────────────────────────────
##
## One global per DataType that serves BOTH systems:
##
##   * GC layout: the mark walker reads the `{elsz, nptrs, offsets[]}`
##     prefix exactly as before (it never reads past the offsets).
##   * Runtime type identity: `obj[-8] & ~15` — the GC header's layout
##     word — IS the Julia type tag. isa / typeassert compare sites
##     (host DataType constants rewritten by
##     `rewrite_host_pointer_constants!`), box type tags
##     (`whisk_typetag_<T>` externs), and `whisk_gc_alloc_typed`
##     headers all converge on this one address per type.
##   * Runtime field access: appended metadata lets
##     `ijl_get_nth_field_checked` walk fields of an `Any` value.
##
## Wire format (16-byte aligned; all words u64 unless noted):
##
##     +0:          elsz                  ┐ layout prefix —
##     +8:          nptrs                 │ mark-walker contract
##     +16:         objectid(T)           │ — FIXED slot: the stable
##                                        │   runtime type id (tag
##                                        │   switches & dispatch keys
##                                        │   load it at tag+16)
##     +24:         offsets[nptrs]        ┘
##     +24+8n:      payload_size          ┐ type metadata
##     +32+8n:      nfields               │
##     +40+8n:      fields_off (from desc │   start; 0 = none)
##     +48+8n:      name bytes + NUL, padded to 8
##     fields_off:  nfields × { u64 offset, u64 field_desc, u64 kind }
##                  kind 0 = pointer slot, 1 = inline (re-box via
##                  field_desc), 2 = unsupported (helper traps)
##
## 16-byte alignment matters: Julia's tag compares mask `& ~15` while
## the GC flags live in bits 0–2 of the header word — both need the
## descriptor's low 4 bits clear.

"""
    whisk_type_descriptor!(mod, T::DataType) → LLVM.GlobalVariable

Get-or-emit the unified type descriptor global for `T` (named
`_whisk_type_<objectid hex>` — stable within a host session, so every
emission path converges on the same global).

Deduplication is BY NAME LOOKUP in the module, never via cached
`GlobalVariable` handles: GPUCompiler's addrspace-removal pass CLONES
every global after `finish_ir!` (handles created there dangle by the
time the post-link pipeline runs — was a use-after-free SIGSEGV), but
names survive the clone.
"""
# Reverse registry: descriptor global name → Julia type. Host-session
# stable (names embed objectid), used by closed-world subtype lowering
# to recover the abstract RHS of `jl_subtype` calls from IR constants.
const DESCRIPTOR_NAME_TO_TYPE = Dict{String, DataType}()

"""
    descriptor_global_type(name) → DataType | nothing

Resolve a descriptor global NAME back to its Julia type. Handles the
`.N` suffix `LLVM.link!` appends when an internal global from a
sub-module collides with one already in the destination.
"""
function descriptor_global_type(name::AbstractString)
    T = get(DESCRIPTOR_NAME_TO_TYPE, name, nothing)
    T !== nothing && return T
    m = match(r"^(.*)\.\d+$", name)
    m === nothing && return nothing
    return get(DESCRIPTOR_NAME_TO_TYPE, m.captures[1], nothing)
end

function whisk_type_descriptor!(mod::LLVM.Module, @nospecialize(T::DataType))
    name = "_whisk_type_" * string(objectid(T); base = 16)
    DESCRIPTOR_NAME_TO_TYPE[name] = T
    haskey(LLVM.globals(mod), name) && return LLVM.globals(mod)[name]

    # Layout prefix — same semantics the shape-keyed descriptors had.
    elsz, offsets = if T <: Memory
        memory_element_layout(T)
    elseif isbitstype(T) || (isconcretetype(T) && Base.datatype_pointerfree(T))
        (UInt64(0), UInt64[])
    else
        (UInt64(0), pointer_offsets(T))
    end

    # Field metadata (fixed-size concrete structs only — Memory,
    # String, Symbol and primitives have no Julia-visible fields the
    # runtime helper should expose).
    frecs = Tuple{UInt64, Union{Nothing, DataType}, UInt64}[]
    if !(T <: Memory) && isconcretetype(T) && !isprimitivetype(T) &&
       T !== Symbol && T !== String
        for i in 1:fieldcount(T)
            ft = fieldtype(T, i)
            off = UInt64(fieldoffset(T, i))
            if ft isa DataType && Base.allocatedinline(ft)
                push!(frecs, (off, ft, UInt64(1)))      # inline — re-box
            elseif ft isa Union && Base.allocatedinline(ft)
                push!(frecs, (off, nothing, UInt64(2))) # inline union — unsupported
            else
                push!(frecs, (off, nothing, UInt64(0))) # pointer slot
            end
        end
    end

    label = codeunits(string(T))
    name_padded = ((length(label) + 1 + 7) ÷ 8) * 8
    n  = length(offsets)
    nf = length(frecs)
    fields_off = nf == 0 ? UInt64(0) : UInt64(24 + 8n + 24 + name_padded)

    i8  = LLVM.Int8Type()
    i64 = LLVM.Int64Type()
    prefix_ty = LLVM.ArrayType(i64, 3 + n)
    meta_ty   = LLVM.ArrayType(i64, 3)
    name_ty   = LLVM.ArrayType(i8, name_padded)
    fields_ty = LLVM.ArrayType(i64, 3nf)
    struct_ty = LLVM.StructType([prefix_ty, meta_ty, name_ty, fields_ty])

    # Create the (named) global BEFORE recursing into field
    # descriptors — the name lookup above is the cycle guard. Inline-
    # field recursion is acyclic anyway (a struct can't inline-contain
    # itself).
    gv = LLVM.GlobalVariable(mod, struct_ty, name)

    prefix = LLVM.Constant[LLVM.ConstantInt(i64, reinterpret(Int64, elsz)),
                           LLVM.ConstantInt(i64, n),
                           LLVM.ConstantInt(i64, reinterpret(Int64, UInt64(objectid(T))))]
    for off in offsets
        push!(prefix, LLVM.ConstantInt(i64, reinterpret(Int64, off)))
    end

    payload_size = (T <: Memory || !isconcretetype(T)) ? UInt64(0) :
                   UInt64(Base.aligned_sizeof(T))
    meta = LLVM.Constant[
        LLVM.ConstantInt(i64, reinterpret(Int64, payload_size)),
        LLVM.ConstantInt(i64, nf),
        LLVM.ConstantInt(i64, reinterpret(Int64, fields_off)),
    ]

    name_consts = LLVM.Constant[LLVM.ConstantInt(i8, 0) for _ in 1:name_padded]
    for (j, bch) in enumerate(label)
        name_consts[j] = LLVM.ConstantInt(i8, Int(bch))
    end

    fwords = LLVM.Constant[]
    for (off, ft, kind) in frecs
        push!(fwords, LLVM.ConstantInt(i64, reinterpret(Int64, off)))
        if ft === nothing
            push!(fwords, LLVM.ConstantInt(i64, 0))
        else
            push!(fwords, LLVM.const_ptrtoint(whisk_type_descriptor!(mod, ft), i64))
        end
        push!(fwords, LLVM.ConstantInt(i64, reinterpret(Int64, kind)))
    end

    fields_init = isempty(fwords) ? LLVM.null(fields_ty) :
                                    LLVM.ConstantArray(i64, fwords)
    LLVM.initializer!(gv, LLVM.ConstantStruct(struct_ty, LLVM.Constant[
        LLVM.ConstantArray(i64, prefix),
        LLVM.ConstantArray(i64, meta),
        LLVM.ConstantArray(i8, name_consts),
        fields_init,
    ]))
    LLVM.linkage!(gv, LLVM.API.LLVMInternalLinkage)
    LLVM.constant!(gv, true)
    LLVM.alignment!(gv, 16)
    LLVM.API.LLVMAppendToUsed(mod, [gv], 1)
    return gv
end

"""
    clear_layout_cache!(mod)

Drop the per-module descriptor cache. Call this when the compile
finishes so we don't hold module references across compiles. Also
emits a summary `@warn` if any alloc sites had to fall back to
layout=0 (couldn't decode the type pointer).
"""
function clear_layout_cache!(mod::LLVM.Module)
    delete!(_descriptor_cache_dict, objectid(mod))
    _emit_layout_fallback_summary!(mod)
    delete!(_layout_fallback_log, objectid(mod))
    let log = get(_reconstruct_fallback_log, objectid(mod), nothing)
        if log !== nothing && !isempty(log)
            @warn "Whisk: $(length(log)) host-pointer constant(s) could not be reconstructed (empty-buffer placeholder used). Comparisons/loads against these read zeros." sample = first(log, min(3, length(log)))
        end
        delete!(_reconstruct_fallback_log, objectid(mod))
    end
end

# Per-module log of alloc sites where _decode_datatype_operand returned
# nothing. Stored as a Vector of (operand_string, as_memory) tuples;
# bounded to keep memory in check on pathological compiles.
const _layout_fallback_log = Dict{UInt64, Vector{Tuple{String, Bool}}}()
const _LAYOUT_FALLBACK_LOG_CAP = 32

function _record_layout_fallback!(mod::LLVM.Module, operand, as_memory::Bool)
    log = get!(() -> Tuple{String, Bool}[], _layout_fallback_log, objectid(mod))
    if length(log) < _LAYOUT_FALLBACK_LOG_CAP
        push!(log, (string(operand), as_memory))
    end
    return
end

function _emit_layout_fallback_summary!(mod::LLVM.Module)
    log = get(_layout_fallback_log, objectid(mod), nothing)
    log === nothing && return
    isempty(log) && return
    n = length(log)
    capped = n == _LAYOUT_FALLBACK_LOG_CAP ? "$(n)+" : "$(n)"
    sample = first(log, min(10, length(log)))
    @warn "Whisk GC: $capped alloc site(s) had untyped allocations (fallback to layout=0; raw-bytes mark). These are usually short-lived sub-compiled helper boxes; if a workload starts trapping with use-after-free symptoms, audit these sites." sample
    return
end

# ── Layout-descriptor type registry ──────────────────────────────────
#
# Per-build registry of types whose LayoutDescriptors must be emitted
# into a wasm binary so host-built images can reference them by tag.
# Defined here rather than in jsinterop/ so compile_to_wasm and the
# layout_addr_table emitter (both in earlier include order) can name
# the type in their signatures.
#
# Each entry carries the (elsz, ptr_offsets) tuple keying Whisk's
# existing descriptor cache — emit shares globals with descriptors
# already produced by gc_alloc lowering for types the compiled
# function internally allocates.

"""
    LayoutTagTable

Registry of (elsz, ptr_offsets) tuples assigning each a stable
type_tag (0..N-1). Host-side image-builders use the tag in
layout_fixups; wasm-side `__whisk_layout_addr(tag)` returns the
corresponding rodata descriptor address at upload time.
"""
mutable struct LayoutTagTable
    keys::Vector{Tuple{UInt64, Vector{UInt64}}}            # by tag index
    types::Vector{Any}                                       # parallel: source DataType
    by_key::Dict{Tuple{UInt64, Vector{UInt64}}, UInt32}
end
LayoutTagTable() = LayoutTagTable(
    Tuple{UInt64, Vector{UInt64}}[],
    Any[],
    Dict{Tuple{UInt64, Vector{UInt64}}, UInt32}(),
)

"Get-or-assign tag for type T; idempotent."
function tag_for!(tt::LayoutTagTable, T::DataType)::UInt32
    key, src = _layout_key(T)
    if haskey(tt.by_key, key)
        return tt.by_key[key]
    end
    push!(tt.keys, key)
    push!(tt.types, src)
    tag = UInt32(length(tt.keys) - 1)
    tt.by_key[key] = tag
    return tag
end

# Map a Julia type to the LayoutDescriptor key the wasm-side cache also uses.
#   * Memory{T}   → (sizeof(elem), per_element_pointer_offsets)
#   * everything else → (0, pointer_offsets(T))
_layout_key(T::DataType) =
    T <: Memory ? (memory_element_layout(T), T) :
                  ((UInt64(0), pointer_offsets(T)), T)
