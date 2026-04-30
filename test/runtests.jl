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
    include("m5_math.jl")          # M5 — sin/cos/sqrt/asin → real haversine
    include("m6_push.jl")          # M6 — push!/pop! via ijl_invoke lowering
    include("m7_marshal.jl")       # M7 — generic Whisk.Marshal across the wasm boundary
    include("m8_growth.jl")        # M8 — large-N growth, struct-with-Vector, slicing, churn
    include("m9_layout.jl")        # M9 — pointer_offsets type-coverage (host-side)
    include("m10_closure.jl")      # M10 — closure auto-marshal via image + setter
    include("m11_image.jl")        # M11 — host image-builder ↔ pointer_offsets agreement
    include("m12_image_gc.jl")     # M12 — image upload + pin + gc_collect e2e (#47)
    include("m13_call_indirect.jl") # M13 — wasm call_indirect smoke test
end
