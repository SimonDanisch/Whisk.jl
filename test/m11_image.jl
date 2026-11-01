## M11 — host-side image-builder layout consistency.
##
## `convert2wasm(value)` encodes `value` into a wasm-resident byte
## image. The encoder must agree byte-for-byte with `pointer_offsets(T)`
## for every encoded object — that's what the wasm GC marker reads at
## collect time. `assert_addr_fixups_match!` enforces this invariant
## per object; these tests exercise the M5-era encoders that were
## suspected for #47 (Dict, BitVector, multi-Vector struct, mixed
## struct), and the assertion fires with a precise diff if anything
## drifts. We don't assert specific tag-types because layout-key
## merging is intentional (two types with the same `(elsz, ptr_offsets)`
## key share one descriptor, GC-correctly).

using Test
using Whisk

# Module-scope structs so their identity is stable across @testset blocks.
struct M11Triple
    a::Vector{Int64}
    b::Vector{UInt32}
    c::Vector{Float32}
end

struct M11Mix
    ids::Vector{Int64}
    lookup::Dict{Int64, UInt32}
    mask::BitVector
end

# `convert2wasm` succeeds → encoder agrees with `pointer_offsets` for
# every emitted object. (The drift assertion runs inside the call.)
function expect_encode_ok(value)
    blob, tags = Whisk.convert2wasm(value)
    @test !isempty(blob)
    @test !isempty(tags.keys)
    return blob, tags
end

@testset "M11 · host image builder ↔ pointer_offsets agreement" begin

    @testset "Dict{Int64, UInt32}" begin
        d = Dict(Int64(1) => UInt32(10), Int64(2) => UInt32(20))
        expect_encode_ok(d)
    end

    @testset "BitVector" begin
        bv = trues(64); bv[3] = false; bv[40] = false
        expect_encode_ok(bv)
    end

    @testset "Immutable struct with multiple bitstype Vectors" begin
        v = M11Triple([Int64(1),Int64(2)],
                      [UInt32(10),UInt32(20),UInt32(30)],
                      Float32[3.14, 2.71])
        expect_encode_ok(v)
    end

    @testset "Struct mixing Dict + BitVector + Vector" begin
        v = M11Mix(
            [Int64(100), Int64(200)],
            Dict(Int64(100) => UInt32(1), Int64(200) => UInt32(2)),
            BitVector([true, false, true, true, false]),
        )
        expect_encode_ok(v)
    end

end
