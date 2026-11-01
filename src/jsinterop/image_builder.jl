## Host-side wasm-memory-image builder.
##
## Goal: from host Julia, produce a byte blob that JS can memcpy into
## wasm linear memory, fix up, and pin — yielding a valid Julia object
## the wasm side recovers via `Whisk.unwrap_value(T, addr)`.
##
## Wire format (all little-endian, all u64 unless noted):
##
##     u32 image_size          // bytes of `data`
##     u32 root_offset         // offset within data of the ROOT object's
##                             // objref (= GC-header end). JS pins
##                             // base + root_offset.
##     u32 n_addr_fixups
##     u32 n_layout_fixups
##     u8  data[image_size]    // raw wasm-memory bytes, addresses written
##                             // as offsets-from-image-base
##     u32 addr_fixups[n_addr_fixups]              // each = offset within
##                                                 // image of an 8-byte
##                                                 // slot holding an
##                                                 // image-internal addr;
##                                                 // JS does +B post-memcpy
##     {u32 offset, u32 type_tag}[n_layout_fixups] // GC-header layout_ptr
##                                                 // slots; JS writes
##                                                 // __whisk_layout_addrs
##                                                 // [type_tag] there
##
## Type tags are stable indices into the per-binary type-tag table
## emitted alongside the wasm. The wasm exports
## `__whisk_layout_addrs() → ptr_to_table_of_u64` which JS reads at
## instantiate time to map tag → descriptor address.
##
## Heap-object layout in wasm linear memory (wasm64):
##
##     [GC header 16B: size:u64, layout_ptr|flags:u64]
##     [object payload, sizeof(T) bytes]
##
## `pointer_from_objref(x)` returns address-of-payload (= GC-header-end).
## Pinning uses the payload address; recovery by `unwrap_value(T, addr)`
## reads `Ref{T}` bytes starting at `addr`. So the root we pin is the
## payload of a `Base.RefValue{T}` whose single pointer field holds the
## actual T's payload — mirroring `pin_value`'s shape exactly.

using Base: aligned_sizeof

# ── Image writer ─────────────────────────────────────────────────────

"""
    ImageWriter

Linear byte buffer + side-tables for fixups. Build the image by
appending heap objects; record each address slot in `addr_fixups` and
each GC-header layout_ptr slot in `layout_fixups`.
"""
mutable struct ImageWriter
    bytes::Vector{UInt8}
    addr_fixups::Vector{UInt32}                           # offsets within bytes
    layout_fixups::Vector{Tuple{UInt32, UInt32}}          # (offset, type_tag)
    tags::LayoutTagTable
    root_offset::UInt32                                   # objref of root, set when ROOT is placed
end
ImageWriter() = ImageWriter(UInt8[], UInt32[], Tuple{UInt32, UInt32}[],
                              LayoutTagTable(), UInt32(0))

"Current size of the image, in bytes."
@inline pos(w::ImageWriter) = UInt32(length(w.bytes))

"Reserve n bytes of zeros, return the start offset."
function _reserve!(w::ImageWriter, n::Integer)::UInt32
    off = pos(w)
    resize!(w.bytes, off + n)
    fill!(@view(w.bytes[off+1:off+n]), 0x00)
    return off
end

"Write `v` (8-byte little-endian) at offset `off`."
function _write_u64!(w::ImageWriter, off::Integer, v::Union{UInt64, Int64})
    GC.@preserve w begin
        unsafe_store!(Ptr{UInt64}(pointer(w.bytes) + off), UInt64(v) % UInt64)
    end
    return
end

"Write arbitrary bitstype value at offset (no padding logic — caller pre-aligns)."
function _write_bits!(w::ImageWriter, off::Integer, x)
    sz = sizeof(x)
    GC.@preserve w begin
        ref = Ref(x)
        GC.@preserve ref begin
            ccall(:memcpy, Ptr{Cvoid},
                  (Ptr{Cvoid}, Ptr{Cvoid}, Csize_t),
                  pointer(w.bytes) + off,
                  Base.unsafe_convert(Ptr{Cvoid}, ref),
                  Csize_t(sz))
        end
    end
    return
end

"""
    emit_gc_header!(w, payload_size, T) → (header_off, objref_off)

Append a 16-byte GC header and reserve `payload_size` bytes of
zeros after it. Records a layout_fixup so JS will write the wasm
descriptor address of `T` into the layout_ptr slot.

Returns (offset of GC header start, offset of payload / objref).
The caller fills the payload by writing at `objref_off`.
"""
function emit_gc_header!(w::ImageWriter, payload_size::Integer, T::DataType)
    header_off = _reserve!(w, 16)
    objref_off = pos(w)
    _reserve!(w, payload_size)
    total_size = UInt64(16 + payload_size)
    # GC header offset 0: total size including header, 8-aligned
    _write_u64!(w, header_off, total_size)
    # GC header offset 8: layout_ptr|flags. JS overwrites at upload via
    # layout_fixups; we leave 0 here as a placeholder.
    tag = tag_for!(w.tags, T)
    push!(w.layout_fixups, (UInt32(header_off + 8), tag))
    return header_off, objref_off
end

"Mark an 8-byte slot at `off` as needing post-memcpy +base relocation."
function emit_addr_fixup!(w::ImageWriter, off::Integer, target_offset::Integer)
    _write_u64!(w, off, UInt64(target_offset))
    push!(w.addr_fixups, UInt32(off))
    return
end

# ── Type-specific encoders ───────────────────────────────────────────
#
# Each `encode!(w, x)` appends a fully-formed heap-object representation
# of `x` and returns its **objref offset** (= GC-header-end). Callers
# place returned offsets into address-fixup slots in their own outer
# objects.
#
# Specializations:
#   * Vector{T<:bitstype}: emits a Memory + Vector pair, three addr
#     fixups (ref.ptr_or_offset, ref.mem, Memory.ptr).
#   * generic T (immutable concrete): emits a single GC-header'd block
#     and walks fields via `emit_inline!`.
#
# **Single source of layout truth.** Both the wasm-side GC marker and
# the host-side encoder must agree byte-for-byte on which payload
# offsets contain managed pointers. `pointer_offsets(T)` (in
# `runtime/layout.jl`) is the canonical answer; the marker reads it
# from a wasm rodata descriptor at GC time, the encoder consults it
# directly here. After encoding, `assert_addr_fixups_match!` checks
# the recorded fixups against `pointer_offsets(T)` — drift fires at
# encode time with a clear diff instead of an OOB trap during
# `whisk_gc_collect`.

"""
    encode!(w, m::Memory{T<:bitstype}) → objref_off

Lay out a `Memory{T}` heap object: 16B GC header, 16B (length, ptr)
header, then `length * sizeof(T)` bytes of inline payload. Returns
the objref offset (= start of the (length, ptr) header).
"""
function encode!(w::ImageWriter, m::Memory{T}) where {T}
    isbitstype(T) || error("encode!(::Memory{T}) currently only supports isbitstype(T); got T=$T")
    n    = length(m)
    elsz = aligned_sizeof(T)
    payload = 16 + n * elsz
    _, mem_objref = emit_gc_header!(w, payload, Memory{T})
    _write_u64!(w, mem_objref, UInt64(n))                    # length
    data_off = mem_objref + UInt32(16)
    emit_addr_fixup!(w, mem_objref + 8, data_off)            # Memory.ptr
    if n > 0
        GC.@preserve m w begin
            ccall(:memcpy, Ptr{Cvoid},
                  (Ptr{Cvoid}, Ptr{Cvoid}, Csize_t),
                  pointer(w.bytes) + data_off,
                  Base.unsafe_convert(Ptr{T}, m),
                  Csize_t(n * elsz))
        end
    end
    return mem_objref
end

"""
    encode!(w, v::Vector{T<:bitstype}) → objref_off

Encode a Vector by piggy-backing on the Memory encoder for its
backing buffer, then emitting the 24B Vector payload (ref.ptr_or_offset,
ref.mem, size[1]) with addr fixups for both pointer slots.
"""
function encode!(w::ImageWriter, v::Vector{T}) where {T}
    isbitstype(T) || error("encode!(::Vector{T}) currently only supports isbitstype(T); got T=$T")
    mem_objref = encode!(w, v.ref.mem)
    data_off   = mem_objref + UInt32(16)   # matches Memory.ptr fixup target

    _, vec_objref = emit_gc_header!(w, sizeof(Vector{T}), Vector{T})
    emit_addr_fixup!(w, vec_objref + 0, data_off)            # ref.ptr_or_offset
    emit_addr_fixup!(w, vec_objref + 8, mem_objref)          # ref.mem
    _write_u64!(w, vec_objref + 16, UInt64(length(v)))       # size[1]
    return vec_objref
end

"""
    encode!(w, x::T) → objref_off

Generic encoder for arbitrary concrete `T`. Allocates a `sizeof(T)`-byte
heap block with a GC header tagged for `T`, then writes `T`'s payload
in place via `emit_inline!`. Used for immutable structs (and Julia
mutable types whose layout we treat as inline-with-pointers, like
`Dict` and `BitVector`) — each managed-pointer field recurses + emits
an address fixup at the field's offset.

The layout descriptor for `T` is selected by `tag_for!`: `pointer_offsets(T)`
gives the per-payload pointer slots the GC marker will trace.
`emit_inline!` consults the same function and asserts the fixups it
records match it byte-for-byte.
"""
function encode!(w::ImageWriter, x::T) where {T}
    isconcretetype(T) || error("encode!: expected concrete type, got $T")
    _, objref = emit_gc_header!(w, sizeof(T), T)
    emit_inline!(w, objref, x)
    return objref
end

# Lay out the bytes of `x::T` IN PLACE at offset `dst` (no GC header — the
# caller already emitted one). Single source of truth: `pointer_offsets(T)`
# decides which payload offsets are managed-pointer slots. Walk fields:
#   * field at a pointer offset      → `encode!` the value, emit addr-fixup
#   * field overlapping a pointer    → recurse via `emit_inline!` (inline
#                                       struct with pointers inside it)
#   * bitstype field                  → `write_bits!`
#   * non-bitstype, no pointers      → singleton field, nothing to write
function emit_inline!(w::ImageWriter, dst::Integer, x::T) where {T}
    if isbitstype(T)
        _write_bits!(w, dst, x)
        return
    end
    isconcretetype(T) || error("emit_inline!: non-concrete type $T")

    fixups_before = length(w.addr_fixups)
    pointer_offs  = pointer_offsets(T)
    nf            = fieldcount(T)

    for i in 1:nf
        ft  = fieldtype(T, i)
        off = UInt64(fieldoffset(T, i))
        # Field's byte span IN its container — not `sizeof(ft)`! Mutable
        # fields are stored as 8-byte pointers regardless of the pointee
        # type's full size. Use the next field's offset (or container
        # size) as the boundary so the pointer-offset filter doesn't
        # sweep neighbouring fields.
        field_end = i < nf ? UInt64(fieldoffset(T, i+1)) : UInt64(sizeof(T))
        fv  = getfield(x, i)
        ptrs_in_field = filter(o -> off <= o < field_end, pointer_offs)

        if isbitstype(ft)
            isempty(ptrs_in_field) ||
                error("emit_inline!: $(T).$(fieldname(T,i))::$ft is bitstype but pointer_offsets says it contains a managed pointer at $(ptrs_in_field). Inconsistent layout.")
            _write_bits!(w, dst + off, fv)
        elseif length(ptrs_in_field) == 1 && first(ptrs_in_field) == off
            # Single managed-pointer slot at this field's offset.
            inner = encode!(w, fv)
            emit_addr_fixup!(w, dst + off, inner)
        elseif !isempty(ptrs_in_field)
            # Non-bitstype field that contains pointers at internal offsets.
            # That's an inline struct field with managed-pointer subfields.
            isconcretetype(ft) && !isabstracttype(ft) ||
                error("emit_inline!: $(T).$(fieldname(T,i))::$ft has internal pointers but isn't a concrete struct (offsets=$ptrs_in_field)")
            emit_inline!(w, dst + off, fv)
        else
            # No bytes to write: field is a singleton type (function with
            # no captures, zero-field struct). The marker won't read
            # this offset since pointer_offsets doesn't list it.
        end
    end

    # ── Single source of truth check ───────────────────────────────
    # The marker walks `pointer_offsets(T)` — every offset must have
    # a matching addr_fixup we just recorded in this object's range.
    assert_addr_fixups_match!(w, T, dst, fixups_before, pointer_offs)
    return
end

# Verify the addr_fixups added during this `emit_inline!` call match
# `pointer_offsets(T)` exactly. Drift here is the structural bug
# class #47 was an instance of — catching it at encode time gives
# a precise error instead of OOB at gc_collect.
#
# Recursive encodes of inner heap objects (e.g. a `Vector` field that
# allocates its own GC-headered Memory) add fixups OUTSIDE the current
# object's byte range. We scope the check to fixups landing within
# `[dst, dst + sizeof(T))` — exactly what the marker will read for
# this object.
function assert_addr_fixups_match!(w::ImageWriter, ::Type{T}, dst::Integer,
                                     fixups_before::Integer,
                                     expected::Vector{UInt64}) where {T}
    payload_end = UInt64(dst) + UInt64(sizeof(T))
    actual = UInt64[]
    for i in (fixups_before+1):length(w.addr_fixups)
        f = UInt64(w.addr_fixups[i])
        UInt64(dst) <= f < payload_end || continue
        push!(actual, f - UInt64(dst))
    end
    sort!(actual)
    expected == actual && return
    error("emit_inline! layout drift for $T:\n" *
          "  pointer_offsets says: $(Int.(expected))\n" *
          "  encoder wrote at:     $(Int.(actual))\n" *
          "  (offsets relative to dst=$dst, payload_end=$(Int(payload_end)))")
end

# ── Top-level convert ────────────────────────────────────────────────

"""
    convert2wasm(value) → (blob::Vector{UInt8}, tags::LayoutTagTable)

Top-level entry. Wraps `value` in a `Base.RefValue{typeof(value)}`
shape (matching `Whisk.pin_value`) so the wasm side recovers via
`unwrap_value(typeof(value), addr)`.

Mutable `T` (Vector, Dict, mutable struct) → RefValue payload is one
8-byte pointer to a separately-allocated heap object. Immutable `T`
(plain struct) → RefValue payload is `sizeof(T)` bytes laid out inline,
because that's how Julia stores `Ref(::immutable)` and what
`unsafe_load(Ptr{Ref{T}})` reads.
"""
function convert2wasm(value)
    w = ImageWriter()
    T = typeof(value)
    RefT = Base.RefValue{T}
    _, ref_objref = emit_gc_header!(w, sizeof(RefT), RefT)
    # `Ref{T}`'s storage convention depends on T:
    #   * mutable T (Vector, Dict, mutable struct, String) → payload is
    #     ONE 8-byte pointer to a separately-allocated heap object.
    #   * immutable T (bitstype OR struct with pointer fields inside) →
    #     payload is sizeof(T) bytes laid out inline; pointer fields
    #     are slots within the inline bytes (NOT a pointer-to-T).
    # `pointer_offsets(RefValue{T}) == [0]` doesn't disambiguate these
    # — both shapes can produce that. We have to ask Julia directly.
    if ismutabletype(T) || T <: Array || T === String
        inner = encode!(w, value)
        emit_addr_fixup!(w, ref_objref + 0, inner)
    else
        emit_inline!(w, ref_objref, value)
    end
    w.root_offset = ref_objref
    return _serialize_image(w), w.tags
end

function _serialize_image(w::ImageWriter)::Vector{UInt8}
    n_addr   = length(w.addr_fixups)
    n_layout = length(w.layout_fixups)
    image_sz = length(w.bytes)
    out_sz   = 16 + image_sz + 4*n_addr + 8*n_layout
    out = Vector{UInt8}(undef, out_sz)
    # Header: image_size, root_offset, n_addr_fixups, n_layout_fixups (each u32, LE)
    GC.@preserve out begin
        p = Ptr{UInt32}(pointer(out))
        unsafe_store!(p,    UInt32(image_sz),  1)
        unsafe_store!(p,    UInt32(w.root_offset), 2)
        unsafe_store!(p,    UInt32(n_addr),    3)
        unsafe_store!(p,    UInt32(n_layout),  4)
        # Image bytes
        ccall(:memcpy, Ptr{Cvoid},
              (Ptr{Cvoid}, Ptr{Cvoid}, Csize_t),
              pointer(out) + 16, pointer(w.bytes), Csize_t(image_sz))
        # addr_fixups: u32 each
        af_p = Ptr{UInt32}(pointer(out) + 16 + image_sz)
        for (i, off) in enumerate(w.addr_fixups)
            unsafe_store!(af_p, UInt32(off), i)
        end
        # layout_fixups: (u32 offset, u32 type_tag) each
        lf_p = Ptr{UInt32}(pointer(out) + 16 + image_sz + 4*n_addr)
        for (i, (off, tag)) in enumerate(w.layout_fixups)
            unsafe_store!(lf_p, UInt32(off), 2*(i-1) + 1)
            unsafe_store!(lf_p, UInt32(tag), 2*(i-1) + 2)
        end
    end
    return out
end
