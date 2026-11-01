## M10 — closure auto-marshaling.
##
## When `compile_to_wasm` is given a callable struct with captured fields
## (a closure), the wasm function naturally takes the closure object as
## its first parameter. The compiler now (a) encodes the closure into a
## standalone image blob, (b) emits a wrapper export that hides the
## closure-pointer arg, (c) adds a `whisk_set_closure_root(i64)` setter.
## The JS side calls `instantiate(wasm, {closureImage, manifest})` and
## the closure is auto-installed before user code runs.

using Test
using Whisk

# Build a callable closure with one captured Int64 field.
function _m10_make_adder(n::Int64)
    @noinline function adder(x::Int64)
        x + n
    end
    return adder
end

@testset "m10 — closure auto-marshal" begin
    add5 = _m10_make_adder(5)
    @test !Base.issingletontype(typeof(add5))
    @test fieldnames(typeof(add5)) === (:n,)

    res = compile_module(add5, Tuple{Int64}; name = "entry")
    @test isfile(res.wasm_path)
    @test !isempty(res.closure_image)            # encoder produced bytes
    @test res.manifest !== nothing               # tags merged + manifest emitted
    @test !isempty(res.manifest.tags)            # closure type contributes ≥1 tag

    # Drive the wasm via whisk.js — exercises the full auto-install path.
    got = run_closure_export(res.wasm_path, res.closure_image, res.manifest,
                              "entry", (7,))
    @test got == 12   # 5 + 7

    # Different capture, same shape. Verifies the per-instance install
    # actually reads the captured field (rather than baking 5 into IR).
    add100 = _m10_make_adder(100)
    res2 = compile_module(add100, Tuple{Int64}; name = "entry")
    got2 = run_closure_export(res2.wasm_path, res2.closure_image, res2.manifest,
                               "entry", (3,))
    @test got2 == 103
end
