## M3 — mark-sweep GC.
##
## We can't directly assert "memory was freed" from a Julia test that
## only sees a return value, so we exercise the GC indirectly:
##
##  1. Allocate a Vector inside a loop, return its size.
##     If allocations leaked across iterations we'd run out of heap;
##     the GC must reclaim each iteration's vector.
##
##  2. Force the GC to run explicitly (when whisk_gc_collect is exposed)
##     and verify we're still functional after — i.e. that mark
##     correctly preserves live roots.

using Test

# Allocate `count` separate Vectors inside a loop, each holding `n`
# Int32s, return the last vector's size. With no GC this would
# eventually OOM (we initial-link with 16MB; ~10k allocations of 100
# floats = ~10MB). We check it just runs without trapping.
function many_allocs(count::Int32)::Int32
    last_len = Int32(0)
    i = Int32(1)
    while i <= count
        v = Vector{Int32}(undef, 100)
        @inbounds v[1] = i
        last_len = Int32(length(v))
        i += Int32(1)
    end
    return last_len
end

@testset "M3 · many allocations don't OOM" begin
    @wasm_check many_allocs(Int32) [
        (Int32(10),)   => 100,
        (Int32(1000),) => 100,
    ] name = "many_allocs"
end

# A function that holds a vector across an inner loop, then returns
# from it. The inner loop's allocations should be reclaimable by GC
# while the outer vector stays live (precise rooting via gc frames).
function outer_inner(n::Int32)::Int32
    outer = Vector{Int32}(undef, Int(n))
    @inbounds outer[1] = Int32(7)
    j = Int32(0)
    while j < Int32(50)
        # tmp pressures the heap; should not displace `outer`
        tmp = Vector{Int32}(undef, 32)
        @inbounds tmp[1] = j
        j += Int32(1)
    end
    return @inbounds outer[1]   # must still be 7
end

@testset "M3 · outer alloc retained across inner-loop allocs" begin
    @wasm_check outer_inner(Int32) [
        (Int32(4),)  => 7,
        (Int32(16),) => 7,
    ] name = "outer_inner"
end

# Explicit-collect roundtrip: run the function, force a collect, run
# again. Both calls must produce the right result, and `whisk_gc_collect`
# itself must be exported (we add `--export=whisk_gc_collect` at link time).
function rt_alloc_first(n::Int32)::Int32
    v = Vector{Int32}(undef, Int(n))
    @inbounds v[1] = Int32(123)
    return @inbounds v[1]
end

@testset "M3 · whisk_gc_collect runs cleanly between calls" begin
    res = compile_module(rt_alloc_first, Tuple{Int32}; name = "rt_alloc_first")
    raw = gc_probe(res.wasm_path, res.entry_name, (Int32(8),))
    parsed = (
        r1 = parse(Int, match(r"\"r1\":(-?\d+)", raw).captures[1]),
        r2 = parse(Int, match(r"\"r2\":(-?\d+)", raw).captures[1]),
    )
    @test parsed.r1 == 123
    @test parsed.r2 == 123
end
