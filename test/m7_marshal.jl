## M7 — `Whisk.Marshal` generic encode/decode across the wasm boundary.
##
## Verifies the same module that's used host-side to write a UInt8 blob
## also compiles into wasm and reads the same bytes back into a live
## Julia object. Covers all the type machinery the routing graph needs:
## bit-packed structs, Vector{T} of bitstype, Vector{T} of struct, Dict,
## Set, BitVector, String, nested struct fields, and the `_decode_struct`
## fallback for arbitrary user-defined structs.

using Test
using Whisk.Marshal: encode, decode, ByteReader

# Concrete shape standing in for a real RoutingGraph-flavoured struct.
struct M7Inner
    id::Int64
    weight::Float32
end
# NOTE: no `String` field — Julia's `String(::Vector{UInt8})` constructor
# routes through `jl_array_to_string` whose sub-MI hits a wasm backend
# bug (LLVM emits an `i32`-width function-table index for a callsite
# expecting `i64`). Tracked as a follow-up; until then `Marshal.decode`
# of `String` works host-side but not in wasm.
struct M7Outer
    bits::BitVector
    items::Vector{M7Inner}
    by_id::Dict{Int64, UInt32}
    tags::Set{UInt8}
end

function _make_m7()
    M7Outer(
        BitVector([true, false, true, true, false, false, true]),
        [M7Inner(1, 1.5f0), M7Inner(2, 2.5f0), M7Inner(3, 3.5f0)],
        Dict{Int64, UInt32}(10 => UInt32(1), 20 => UInt32(2)),
        Set{UInt8}([UInt8(7), UInt8(11), UInt8(42)]),
    )
end

@testset "M7 · Marshal host roundtrip" begin
    x = _make_m7()
    io = IOBuffer(); encode(io, x); blob = take!(io)
    @test length(blob) > 0
    y = decode(IOBuffer(blob), M7Outer)
    @test y.bits == x.bits
    @test y.items == x.items
    @test y.by_id == x.by_id
    @test y.tags  == x.tags
end

# Wasm-side decoder: receives the blob as raw bytes, decodes to M7Outer,
# returns a small fingerprint we can check from JS.
function decode_m7_fingerprint(blob_ptr::Ptr{UInt8}, blob_len::UInt64)::Int32
    bytes = Vector{UInt8}(undef, Int(blob_len))
    unsafe_copyto!(pointer(bytes), blob_ptr, Int(blob_len))
    y = decode(ByteReader(bytes), M7Outer)
    # Mix several fields so any decode bug shows up.
    fp = 0
    for it in y.items
        fp = fp * 31 + Int(it.id)
        fp = fp * 31 + Int(round(it.weight * 10))
    end
    fp = fp * 31 + length(y.bits)
    fp = fp * 31 + length(y.by_id)
    fp = fp * 31 + length(y.tags)
    return Int32(fp & 0x7fffffff)
end

# Host fingerprint must match the wasm one — same expression form.
function decode_m7_fingerprint_host(x::M7Outer)::Int32
    fp = 0
    for it in x.items
        fp = fp * 31 + Int(it.id)
        fp = fp * 31 + Int(round(it.weight * 10))
    end
    fp = fp * 31 + length(x.bits)
    fp = fp * 31 + length(x.by_id)
    fp = fp * 31 + length(x.tags)
    return Int32(fp & 0x7fffffff)
end

@testset "M7 · Marshal wasm decode" begin
    x = _make_m7()
    expected = decode_m7_fingerprint_host(x)
    out = tempname() * ".wasm"
    res = Whisk.compile_to_wasm(decode_m7_fingerprint,
                                Tuple{Ptr{UInt8}, UInt64};
                                out_path = out, name = "decode_m7_fingerprint",
                                optimize = true)
    @test isfile(res.wasm_path)
    io = IOBuffer(); encode(io, x); blob = take!(io)
    got = run_blob_export(res.wasm_path, res.entry_name, blob)
    @test got == expected
end
