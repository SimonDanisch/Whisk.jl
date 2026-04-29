## Generic encode/decode for the wasm boundary
##
## One source of truth, loaded both server-side (where it writes blobs)
## and into Whisk-compiled wasm modules (where it reads them back into
## live Julia objects). Multiple dispatch + inference do the
## specialization — every `encode(io, ::RoutingGraph)` call site is
## inlined into a flat sequence of writes; same for `decode`.
##
## Covered: `isbitstype` primitives & bit-packed structs, `Vector{T}`,
## `Dict{K,V}`, `Set{T}`, `BitVector`, `String`, arbitrary concrete
## structs / Tuples / NamedTuples (via the `_decode_struct` fallback).
##
## Not covered (errors at encode time): abstract-typed fields,
## mutable-struct identity / aliasing, function/IO/pointer fields. The
## first is a v2 dynamic-dispatch concern.

module Marshal

export encode, decode, ByteWriter, RawByteWriter, ByteReader

# ── Raw-buffer cursors ──────────────────────────────────────────────
# `IOBuffer` works host-side but pulls `ijl_alloc_string` and friends
# into wasm via `take!` / construction paths. Whisk's wasm backend
# can't lower the variadic indirect calls those produce. So Marshal
# operates on a thin `Vector{UInt8}` + `Int` cursor that has no Julia
# runtime dependencies.
#
# Both readers and writers expose `read(::Type)` / `write(value)`
# methods — the same surface area `IOBuffer` does for the encoders
# below — but the implementation is hand-rolled, branchless, and
# free of Julia-internal calls.

mutable struct ByteWriter
    data::Vector{UInt8}
    pos::Int
end

"""
    ByteWriter(capacity = 0) → ByteWriter

A growable byte buffer, but the caller is expected to pre-size it to
the upper bound of what they'll write. Whisk-compiled code should
never let the buffer grow at runtime: `Vector{UInt8}` resize routes
through `jl_alloc_genericmemory_unchecked` via `ijl_load_and_lookup`,
which Whisk traps as unrepresentable in wasm. Use a generous initial
capacity instead.
"""
ByteWriter(capacity::Integer = 0) = ByteWriter(Vector{UInt8}(undef, Int(capacity)), 0)

@inline function _write_bitstype!(w::ByteWriter, x::T) where T
    n = sizeof(T)
    @assert w.pos + n <= length(w.data) "ByteWriter overflow: pre-size with a larger capacity"
    GC.@preserve w begin
        ptr = pointer(w.data, w.pos + 1)
        unsafe_store!(reinterpret(Ptr{T}, ptr), x)
    end
    w.pos += n
    return
end

@inline function _write_bytes!(w::ByteWriter, src_ptr::Ptr{UInt8}, n::Integer)
    @assert w.pos + n <= length(w.data) "ByteWriter overflow: pre-size with a larger capacity"
    GC.@preserve w begin
        unsafe_copyto!(pointer(w.data, w.pos + 1), src_ptr, n)
    end
    w.pos += n
    return
end

# Final byte slice — `view` over the underlying Vector to avoid a
# `resize!` (which would trip the `jl_alloc_genericmemory_unchecked`
# trap inside wasm).
take_bytes(w::ByteWriter) = view(w.data, 1:w.pos)

"""
    RawByteWriter(base::Ptr{UInt8}, capacity::Integer) → RawByteWriter

Like `ByteWriter` but writes directly into a caller-provided
`Ptr{UInt8}` buffer — no intermediate `Vector{UInt8}` allocation, no
final memcpy. The hot-path writer for wasm entry points where JS
provides the output buffer in linear memory and the caller wants
to skip the round-trip through a heap-allocated staging Vector.

The caller is responsible for bounds (compute `encoded_size(x)`
upfront and verify it fits in `capacity` before writing). Overflow
asserts.
"""
mutable struct RawByteWriter
    base::Ptr{UInt8}
    pos::Int
    cap::Int
end
RawByteWriter(base::Ptr{UInt8}, capacity::Integer) =
    RawByteWriter(base, 0, Int(capacity))

@inline function _write_bitstype!(w::RawByteWriter, x::T) where T
    n = sizeof(T)
    @assert w.pos + n <= w.cap "RawByteWriter overflow: caller must pre-check via encoded_size"
    unsafe_store!(reinterpret(Ptr{T}, w.base + w.pos), x)
    w.pos += n
    return
end

@inline function _write_bytes!(w::RawByteWriter, src_ptr::Ptr{UInt8}, n::Integer)
    @assert w.pos + n <= w.cap "RawByteWriter overflow: caller must pre-check via encoded_size"
    unsafe_copyto!(w.base + w.pos, src_ptr, n)
    w.pos += n
    return
end

mutable struct ByteReader
    data::Vector{UInt8}
    pos::Int
end
ByteReader(data::Vector{UInt8}) = ByteReader(data, 0)

@inline function _read_bitstype(r::ByteReader, ::Type{T}) where T
    n = sizeof(T)
    @assert r.pos + n <= length(r.data) "ByteReader: read past end"
    GC.@preserve r begin
        ptr = pointer(r.data, r.pos + 1)
        v = unsafe_load(reinterpret(Ptr{T}, ptr))
    end
    r.pos += n
    return v
end

@inline function _read_bytes!(r::ByteReader, dst_ptr::Ptr{UInt8}, n::Integer)
    @assert r.pos + n <= length(r.data) "ByteReader: read past end"
    GC.@preserve r begin
        unsafe_copyto!(dst_ptr, pointer(r.data, r.pos + 1), n)
    end
    r.pos += n
    return
end

# ── encode ───────────────────────────────────────────────────────────
# Two surfaces:
#   * `encode(io::IO, x)` — host-side convenience, used by SimiMaps's
#     `serialize_graph` etc. Forwards to a `ByteWriter` then writes
#     the resulting bytes once.
#   * `encode(w::ByteWriter, x)` — the wasm-friendly path, used by
#     compiled code on both sides. No Julia-runtime calls.

# Host-only entry. Walks the value once to compute the encoded size,
# then encodes into a tightly-fitting `ByteWriter` (no `resize!` —
# that path traps in wasm). Host code never enters `ByteWriter`'s
# growth path either, so the same code runs in both worlds.
function encode(io::IO, x)
    cap = encoded_size(x)
    w = ByteWriter(cap)
    encode(w, x)
    write(io, view(w.data, 1:w.pos))
    return
end

# Quick pre-pass that returns the byte length `encode(w, x)` will
# write. Mirrors the dispatch shape of `encode` so adding a new type
# means adding one method here too. For arbitrary structs we recurse
# field-by-field via `_struct_encoded_size`.
encoded_size(x::T) where T = isbitstype(T) ? sizeof(T) : _struct_encoded_size(x)
encoded_size(v::Vector{T}) where T = 4 + (isbitstype(T) ?
    length(v) * sizeof(T) : sum(encoded_size, v; init = 0))
encoded_size(d::Dict{K,V}) where {K,V} =
    4 + sum(((k, v),) -> encoded_size(k) + encoded_size(v), d; init = 0)
encoded_size(s::Set) = 4 + sum(encoded_size, s; init = 0)
encoded_size(bv::BitVector) = 8 + length(bv.chunks) * sizeof(UInt64)
encoded_size(s::String) = 4 + sizeof(s)

@generated function _struct_encoded_size(x::T) where T
    if fieldcount(T) == 0
        return :(0)
    end
    expr = :(encoded_size(getfield(x, 1)))
    for i in 2:fieldcount(T)
        expr = :($expr + encoded_size(getfield(x, $i)))
    end
    return expr
end

# Encode methods accept either `ByteWriter` (Vector-backed) or
# `RawByteWriter` (raw pointer). The shared `_write_bitstype!` /
# `_write_bytes!` helpers handle the storage difference; the encode
# logic itself doesn't care.
const AnyByteWriter = Union{ByteWriter, RawByteWriter}

encode(w::AnyByteWriter, x::T) where T = isbitstype(T) ?
    _write_bitstype!(w, x) : _encode_struct(w, x)

function encode(w::AnyByteWriter, v::Vector{T}) where T
    _write_bitstype!(w, UInt32(length(v)))
    if isbitstype(T)
        n = length(v) * sizeof(T)
        n == 0 && return
        GC.@preserve v begin
            _write_bytes!(w, reinterpret(Ptr{UInt8}, pointer(v)), n)
        end
    else
        for el in v
            encode(w, el)
        end
    end
    return
end

function encode(w::AnyByteWriter, d::Dict{K,V}) where {K,V}
    _write_bitstype!(w, UInt32(length(d)))
    for (k, v) in d
        encode(w, k); encode(w, v)
    end
    return
end

function encode(w::AnyByteWriter, s::Set{T}) where T
    _write_bitstype!(w, UInt32(length(s)))
    for k in s
        encode(w, k)
    end
    return
end

function encode(w::AnyByteWriter, bv::BitVector)
    _write_bitstype!(w, UInt64(length(bv)))
    chunks = bv.chunks
    n = length(chunks) * sizeof(UInt64)
    n == 0 && return
    GC.@preserve chunks begin
        _write_bytes!(w, reinterpret(Ptr{UInt8}, pointer(chunks)), n)
    end
    return
end

function encode(w::AnyByteWriter, s::String)
    n = sizeof(s)
    _write_bitstype!(w, UInt32(n))
    n == 0 && return
    GC.@preserve s begin
        _write_bytes!(w, pointer(s), n)
    end
    return
end

# `Symbol` is opaque (zero fields, not bitstype), so without an explicit
# method the struct fallback would happily encode 0 bytes per symbol and
# the JS side would receive nameless garbage. Treat them as length-prefixed
# UTF-8 names — same wire format as `String`, so the JS encoder can produce
# blobs read back by `decode(::ByteReader, Symbol)` round-trip-clean.
encoded_size(s::Symbol) = 4 + sizeof(String(s))

function encode(w::AnyByteWriter, s::Symbol)
    name = String(s)
    encode(w, name)
    return
end

# Same shape as `_decode_struct`: unroll at compile time so each
# `encode(io, getfield(x, i))` call site knows the concrete field type
# and dispatches statically. A runtime `for i in 1:fieldcount(T)` loop
# falls back to dynamic dispatch (every field a `jl_apply_generic`),
# which LLVM-wasm can't lower (`getAlignment` segfault during ISel).
@generated function _encode_struct(w, x::T) where T
    body = Expr(:block)
    for i in 1:fieldcount(T)
        push!(body.args, :(encode(w, getfield(x, $i))))
    end
    push!(body.args, :(return))
    body
end

# ── decode ───────────────────────────────────────────────────────────
# Mirror surface of `encode`: host-friendly `IO` overload that buffers
# into a `ByteReader`, plus the wasm-friendly `ByteReader` path.

function decode(io::IO, ::Type{T}) where T
    bytes = read(io)
    return decode(ByteReader(bytes), T)
end

decode(r::ByteReader, ::Type{T}) where T = isbitstype(T) ?
    _read_bitstype(r, T) : _decode_struct(r, T)

function decode(r::ByteReader, ::Type{Vector{T}}) where T
    n = Int(_read_bitstype(r, UInt32))
    v = Vector{T}(undef, n)
    if isbitstype(T)
        n == 0 && return v
        GC.@preserve v begin
            _read_bytes!(r, reinterpret(Ptr{UInt8}, pointer(v)), n * sizeof(T))
        end
    else
        @inbounds for i in 1:n
            v[i] = decode(r, T)
        end
    end
    return v
end

function decode(r::ByteReader, ::Type{Dict{K,V}}) where {K,V}
    n = Int(_read_bitstype(r, UInt32))
    d = Dict{K,V}(); sizehint!(d, n)
    for _ in 1:n
        k = decode(r, K)
        d[k] = decode(r, V)
    end
    return d
end

function decode(r::ByteReader, ::Type{Set{T}}) where T
    n = Int(_read_bitstype(r, UInt32))
    s = Set{T}(); sizehint!(s, n)
    for _ in 1:n
        push!(s, decode(r, T))
    end
    return s
end

function decode(r::ByteReader, ::Type{BitVector})
    n = Int(_read_bitstype(r, UInt64))
    bv = BitVector(undef, n)
    chunks = bv.chunks
    nbytes = length(chunks) * sizeof(UInt64)
    nbytes == 0 && return bv
    GC.@preserve chunks begin
        _read_bytes!(r, reinterpret(Ptr{UInt8}, pointer(chunks)), nbytes)
    end
    return bv
end

# Strings still go through `String(::Vector{UInt8})`, which hits the
# wasm32/64 ABI bug (#20). Use only host-side; wasm code that needs
# string fields should encode/decode the raw byte vector and let the
# JS side stringify.
function decode(r::ByteReader, ::Type{String})
    n = Int(_read_bitstype(r, UInt32))
    bytes = Vector{UInt8}(undef, n)
    n == 0 && return String(bytes)
    GC.@preserve bytes begin
        _read_bytes!(r, pointer(bytes), n)
    end
    return String(bytes)
end

decode(r::ByteReader, ::Type{Symbol}) = Symbol(decode(r, String))

# Struct fallback is the one place we need a fixed-shape unroll: no
# ntuple/splat (those route through a `rethrow` that the wasm backend
# can't lower) and no runtime field walk (constructor wants a fixed
# arity). `@generated` here just expands `T(decode(io, F1), ..., decode(io, FN))`
# at compile time — multiple dispatch handles each `decode(io, Fi)`.
@generated function _decode_struct(r, ::Type{T}) where T
    args = [:(decode(r, $(fieldtype(T, i)))) for i in 1:fieldcount(T)]
    Expr(:call, T, args...)
end

end # module
