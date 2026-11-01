## M9 — pointer_offsets / memory_element_layout coverage.
##
## Host-side unit tests for the type introspection that drives
## Whisk's precise GC. A misclassification here = silent live-data
## reclamation or pointer-following garbage at runtime, both of
## which are nightmare-grade bugs. Cover every Julia type shape
## that real workloads hit — immutable structs with refs, nested
## immutables, NamedTuples, Tuples, Dicts, Strings, Symbols,
## recursive types — so changes that break classification flip a
## red test instead of a runtime trap weeks later.

using Test
using Whisk: pointer_offsets, memory_element_layout

# ── 1. Bitstype — empty offsets ─────────────────────────────────────

@testset "M9 · bitstypes have empty pointer_offsets" begin
    @test pointer_offsets(Float32) == UInt64[]
    @test pointer_offsets(Int64)   == UInt64[]
    @test pointer_offsets(Bool)    == UInt64[]
    @test pointer_offsets(UInt8)   == UInt64[]
    @test pointer_offsets(Float64) == UInt64[]
end

# ── 2. Tuple of bitstypes — empty offsets ───────────────────────────

@testset "M9 · Tuple of bitstypes is pointer-free" begin
    @test pointer_offsets(Tuple{Int32, Int32}) == UInt64[]
    @test pointer_offsets(Tuple{Float64, Int, UInt8}) == UInt64[]
    @test pointer_offsets(NTuple{4, Float32}) == UInt64[]
end

# ── 3. NamedTuple of bitstypes — empty offsets ──────────────────────

@testset "M9 · NamedTuple of bitstypes is pointer-free" begin
    nt_ty = typeof((a = 1, b = 2.5f0))
    @test pointer_offsets(nt_ty) == UInt64[]
end

# ── 4. Single-Vector struct ─────────────────────────────────────────

struct OneVec
    v::Vector{Float32}
end

@testset "M9 · single Vector field at offset 0" begin
    @test pointer_offsets(OneVec) == UInt64[0]
end

# ── 5. Mixed bitstype + Vector — Vector pointer at the right offset

struct M9_Tile
    id::Int32                 # offset 0, 4 bytes; padding to offset 8
    points::Vector{Float32}   # offset 8
end

@testset "M9 · Tile-shape struct: ptr at offset 8" begin
    @test pointer_offsets(M9_Tile) == UInt64[8]
end

# ── 6. Nested immutable struct with refs — offsets are absolute ────

struct M9_Inner
    x::Int            # offset 0
    y::Vector{Int}    # offset 8
end

struct M9_Outer
    a::Int            # offset 0
    b::M9_Inner       # offset 8 (Inner is 16 bytes inline: x + y_ptr)
end

@testset "M9 · nested immutable: offsets are absolute" begin
    @test pointer_offsets(M9_Outer) == UInt64[16]   # 8 (offset of b) + 8 (offset of y within Inner)
end

# ── 7. Multiple pointer fields ─────────────────────────────────────

struct M9_TwoVec
    a::Vector{Int}    # offset 0
    b::Vector{Int}    # offset 8
end

@testset "M9 · two Vector fields" begin
    @test pointer_offsets(M9_TwoVec) == UInt64[0, 8]
end

# ── 8. String field is a reference ─────────────────────────────────

struct M9_Named
    name::String
    age::Int
end

@testset "M9 · String field is treated as a reference" begin
    offs = pointer_offsets(M9_Named)
    # name should be at offset 0; age (Int) is bitstype.
    @test 0 in offs
    @test length(offs) == 1
end

# ── 9. Symbol field is a reference ─────────────────────────────────

struct M9_Sym
    label::Symbol
    flag::Bool
end

@testset "M9 · Symbol field is treated as a reference" begin
    @test 0 in pointer_offsets(M9_Sym)
end

# ── 10. Memory{T}: bitstype T → no per-element scan ────────────────

@testset "M9 · Memory{Float32}: bitstype, no scan" begin
    elsz, offs = memory_element_layout(Memory{Float32})
    @test elsz == 4
    @test isempty(offs)
end

@testset "M9 · Memory{Int64}: bitstype, no scan" begin
    elsz, offs = memory_element_layout(Memory{Int64})
    @test elsz == 8
    @test isempty(offs)
end

# ── 11. Memory{Vector{T}}: boxed elements, 8-byte slots, ptr at 0 ─

@testset "M9 · Memory{Vector{Float32}}: boxed elements" begin
    elsz, offs = memory_element_layout(Memory{Vector{Float32}})
    @test elsz == 8
    @test offs == UInt64[0]
end

# ── 12. Memory{Tile}: inline-stored struct with Vector field ───────

@testset "M9 · Memory{Tile}: inline, Vector ptr at offset 8 per element" begin
    elsz, offs = memory_element_layout(Memory{M9_Tile})
    @test elsz == 16   # Int32 padded to 8 + Vector ptr 8
    @test offs == UInt64[8]
end

# ── 13. Memory{String}: boxed (String is a reference type) ─────────

@testset "M9 · Memory{String}: boxed elements" begin
    elsz, offs = memory_element_layout(Memory{String})
    @test elsz == 8
    @test offs == UInt64[0]
end

# ── 14. Tuple{Int, Vector} — pointer at the second slot ────────────

@testset "M9 · Tuple{Int, Vector{Int}}: ptr at offset 8" begin
    @test pointer_offsets(Tuple{Int, Vector{Int}}) == UInt64[8]
end

# ── 15. NamedTuple with mixed kinds ─────────────────────────────────

@testset "M9 · NamedTuple with mixed kinds" begin
    nt_ty = typeof((id = 1, name = "x", v = Int[]))
    offs = pointer_offsets(nt_ty)
    # id at offset 0 (bitstype), name at offset 8 (String), v at offset 16 (Vector)
    @test 8 in offs
    @test 16 in offs
    @test 0 ∉ offs   # id is bitstype, not a pointer slot
    @test length(offs) == 2
end

# ── 16. Recursive type via reference field — terminates ────────────
# A struct that holds a Vector of itself. The Vector pointer at offset
# 0 is a real heap pointer; pointer_offsets must not infinitely
# recurse trying to inline the Vector's element type.

struct M9_Tree
    children::Vector{M9_Tree}
end

@testset "M9 · recursive type — does not infinite-loop" begin
    @test pointer_offsets(M9_Tree) == UInt64[0]
end

# ── 17. Dict{K,V}: three Memory pointers at offsets 0, 8, 16 ───────
# Layout depends on Julia version; assert the structural property
# (the first three fields are Memory references) without hardcoding
# offsets that could shift. As of Julia 1.12 they're at 0/8/16.

@testset "M9 · Dict{UInt8, Float32}: Memory refs in low slots" begin
    offs = pointer_offsets(Dict{UInt8, Float32})
    @test 0 in offs
    @test 8 in offs
    @test 16 in offs
    @test length(offs) >= 3
end

# ── 18. Pointer-free immutable: no pointer offsets ─────────────────

struct M9_Bits
    a::Int32
    b::Int32
    c::Float32
end

@testset "M9 · pointer-free immutable struct: empty" begin
    @test pointer_offsets(M9_Bits) == UInt64[]
end

# ── 19. Bool/Char/UInt8 special values are bitstype ────────────────

@testset "M9 · Bool/Char/UInt8 are bitstype" begin
    @test pointer_offsets(Char) == UInt64[]
    struct_ty = Tuple{Bool, Char, UInt8}
    @test pointer_offsets(struct_ty) == UInt64[]
end

# ── 20. Vector{Vector{T}} layered references ──────────────────────

@testset "M9 · Vector{Vector{Int}} as a struct field is one ptr" begin
    struct_ty = Tuple{Int, Vector{Vector{Int}}}
    @test pointer_offsets(struct_ty) == UInt64[8]
end

# ── 21. Function-typed field is a reference ────────────────────────

struct M9_WithFn
    cb::Function
    n::Int
end

@testset "M9 · Function field is treated as a reference" begin
    @test 0 in pointer_offsets(M9_WithFn)
end
