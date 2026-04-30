## M6 — `push!` / `pop!` and the `ijl_invoke` lowering pipeline that
## makes them work. Exercises:
##   * `lower_invokes!` — replace `ijl_invoke` of statically-resolved
##     MethodInstances with direct calls to sub-compiled bodies.
##   * sub-module addrspace stripping (`RemoveJuliaAddrspacesPass`) —
##     so the wasm backend never sees `ptr addrspace(10)` (= externref).
##   * `ijl_box_*` runtime — boxes Int64 / Int32 / etc. via
##     `whisk_gc_alloc` from inside exception-construction call frames.
##
## These together unblock `BinaryMinHeap` and the real `SimiMaps.astar`.

using Test

# Smallest push! shape: empty Vector{Int32}, push one element, read it.
function push_int_vec(::Int32)::Int32
    v = Int32[]
    push!(v, Int32(7))
    return @inbounds v[1]
end

@testset "M6 · push! Vector{Int32}, single element" begin
    res = @wasm_check push_int_vec(Int32) [
        (Int32(0),) => 7,
    ] name = "push_int_vec"
    @test !occursin(r"(?m)^declare .*@i?jl_invoke\(", res.llvm_ir)
    @test !occursin("addrspace(10)", res.llvm_ir)
end

# Multiple push!es in a row — exercises `_growend!` more than once and
# verifies the second push doesn't trample the first.
function push_three(::Int32)::Int32
    v = Int32[]
    push!(v, Int32(11))
    push!(v, Int32(22))
    push!(v, Int32(33))
    @inbounds v[1] + v[2] + v[3]
end

@testset "M6 · push! three Int32 elements, sum" begin
    @wasm_check push_three(Int32) [
        (Int32(0),) => 11 + 22 + 33,
    ] name = "push_three"
end

# Float payload — distinct sub-MI path than Int32 (different `_growend!`
# specialization), but sub-compile + addrspace strip + box should
# handle it identically.
function push_float(::Int32)::Float32
    v = Float32[]
    push!(v, 1.5f0)
    push!(v, 2.5f0)
    @inbounds v[1] + v[2]
end

@testset "M6 · push! Vector{Float32}" begin
    @wasm_check push_float(Int32) [
        (Int32(0),) => Float32(4.0),
    ] name = "push_float"
end
