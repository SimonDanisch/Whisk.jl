## M8 — worst-case allocator + GC stress.
##
## `m6_push.jl` covered the syntactic shape of `push!` (3 elements,
## bitstypes only). It missed a real-graph routing run because three
## pushes can stay within Vector's initial capacity (no actual
## `_growend!` call), and because struct-with-`Vector` fields go
## through different specialisations than primitive Int32.
##
## This file exercises the paths a non-trivial workload actually
## hits: thousands of elements (forcing several reallocations),
## struct-shaped payloads, slicing, two-pass build patterns, and
## sustained alloc/collect cycles.

using Test

# ── 1. push! enough that the underlying buffer ACTUALLY reallocates ──

function push_n_int32(n::Int32)::Int32
    v = Int32[]
    for i in Int32(1):n
        push!(v, i)
    end
    s = Int32(0)
    @inbounds for i in 1:length(v)
        s += v[i]
    end
    s
end

@testset "M8 · push! 1000 Int32 — sum = 500500" begin
    @wasm_check push_n_int32(Int32) [
        (Int32(1000),) => Int32(sum(1:1000)),  # 500500
    ] name = "push_n_int32"
end

@testset "M8 · push! 5000 Int32 — multiple realloc rounds" begin
    @wasm_check push_n_int32(Int32) [
        (Int32(5000),) => Int32(sum(1:5000)),  # 12502500
    ] name = "push_n_int32_5000"
end

# ── 2. push! Float32 at scale ─────────────────────────────────────────

function push_n_float(n::Int32)::Float32
    v = Float32[]
    for i in 1:Int(n)
        push!(v, Float32(i) * 0.5f0)
    end
    s = 0f0
    @inbounds for i in 1:length(v)
        s += v[i]
    end
    s
end

@testset "M8 · push! 2000 Float32 sum" begin
    expected = Float32(sum(Float32(i) * 0.5f0 for i in 1:2000))
    @wasm_check push_n_float(Int32) [
        (Int32(2000),) => expected,
    ] name = "push_n_float_2000"
end

# ── 3. Vector slicing on a primitive-element Vector ──────────────────
# `v[1:n]` routes through `getindex(::Vector, ::UnitRange)` →
# `similar` + `copyto!`. For non-bitstype element types this hits
# `_growend!`-adjacent code paths. For primitives it should be a
# straight memcpy.

function slice_range_sum(n::Int32)::Int32
    full = Int32[]
    for i in Int32(1):n
        push!(full, i)
    end
    half = full[1:div(Int(n), 2)]
    s = Int32(0)
    @inbounds for i in 1:length(half)
        s += half[i]
    end
    s
end

@testset "M8 · Vector{Int32}[1:n/2] — slicing works" begin
    @wasm_check slice_range_sum(Int32) [
        (Int32(100),) => Int32(sum(1:50)),   # 1275
        (Int32(1000),) => Int32(sum(1:500)),  # 125250
    ] name = "slice_range_sum"
end

# ── 4. push! a struct that contains a Vector field ───────────────────
# This is the shape that build_route_segments actually emits: a
# RouteSegment-like struct whose `coords` field is itself a `Vector`.
# Different `_growend!` specialisation than primitive elements;
# also exercises GC marking through the nested Vector.

struct Tile
    id::Int32
    points::Vector{Float32}
end

function build_tiles(n::Int32)::Float32
    tiles = Tile[]
    for i in Int32(1):n
        pts = Float32[]
        for j in 1:5
            push!(pts, Float32(i) + Float32(j) * 0.1f0)
        end
        push!(tiles, Tile(i, pts))
    end
    # Sum the first point of each tile — verifies both the outer
    # Vector{Tile} and the inner Vector{Float32} survived.
    s = 0f0
    @inbounds for k in 1:length(tiles)
        s += tiles[k].points[1]
    end
    s
end

@testset "M8 · push!(Vector{Tile}, struct-with-Vector)" begin
    expected_500 = Float32(sum(Float32(i) + 0.1f0 for i in 1:500))
    @wasm_check build_tiles(Int32) [
        (Int32(500),) => expected_500,
    ] name = "build_tiles_500"
end

# ── 5. Two-pass build (count → preallocate → fill) ───────────────────
# Mirrors the `_reconstruct` pattern in SimiMaps' astar — the safe
# alternative to `push!` when you can compute the length upfront.
# This SHOULD work even without `_growend!` lowered, because we
# only allocate one exact-size Vector.

function two_pass_count_then_fill(limit::Int32)::Int32
    n = 0
    @inbounds for i in 1:Int(limit)
        n += isodd(i) ? 1 : 0
    end
    out = Vector{Int32}(undef, n)
    j = 0
    @inbounds for i in 1:Int(limit)
        if isodd(i)
            j += 1
            out[j] = Int32(i)
        end
    end
    s = Int32(0)
    @inbounds for k in 1:length(out)
        s += out[k]
    end
    s
end

@testset "M8 · two-pass count + preallocate + fill" begin
    @wasm_check two_pass_count_then_fill(Int32) [
        (Int32(1000),) => Int32(sum(1:2:999)),  # 250000
    ] name = "two_pass_1000"
end

# ── 6. Sustained alloc + collect cycles ──────────────────────────────
# Allocate a fresh Vector inside a loop so each iteration's previous
# Vector becomes unreachable. Without a working collect, the bump
# heap would grow unbounded and `whisk_ensure_capacity` would
# eventually grow past `--max-memory`. The wasm wrapper in
# `utils.jl` initialises memory to a small size, so this exercises
# the grow-and-collect interplay.

function alloc_churn(n::Int32)::Int32
    last_first = Int32(0)
    @inbounds for k in 1:Int(n)
        v = Int32[]
        for i in Int32(1):Int32(100)
            push!(v, i + Int32(k))
        end
        last_first = v[1]
    end
    last_first
end

@testset "M8 · 200×100-element churn — no leak / no trap" begin
    @wasm_check alloc_churn(Int32) [
        (Int32(200),) => Int32(201),   # last v[1] = 1 + 200
    ] name = "alloc_churn_200"
end
