## Whisk regression suite.
##
## Each milestone has its own file. Currently-broken features sit as
## `@test_broken` so we see them flip green when implemented.

using Test

include("utils.jl")

@testset "Whisk" begin
    include("m1_scalar.jl")        # M1 — scalar arithmetic, no allocator
    include("m2_allocations.jl")   # M2 — Vector / String (broken until M2)
    include("m3_gc.jl")            # M3 — mark-sweep GC across loops
    include("m4_dict.jl")          # M4 — Dict, Set, BitVector
end
