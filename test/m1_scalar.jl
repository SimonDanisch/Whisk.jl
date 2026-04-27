## M1 — scalar arithmetic, no allocator, no GC.
##
## What we pin here MUST keep working in M2/M3/M4 — these are the
## simplest passes and any regression here means the pipeline itself
## broke.

using Test

inc(x::Int32) = x + Int32(1)

@testset "M1 · Int32 arithmetic" begin
    @wasm_check inc(Int32) [
        (Int32(0),)  => 1,
        (Int32(41),) => 42,
        (Int32(-1),) => 0,
    ] name = "inc"
end

# Define helpers at module scope so the compiler can see them.
add_i32(a::Int32, b::Int32) = a + b
mul_i32(a::Int32, b::Int32) = a * b
sub_i32(a::Int32, b::Int32) = a - b

@testset "M1 · Int32 binary ops" begin
    @wasm_check add_i32(Int32, Int32) [
        (Int32(2), Int32(3))   => 5,
        (Int32(-7), Int32(7))  => 0,
    ] name = "add_i32"

    @wasm_check mul_i32(Int32, Int32) [
        (Int32(6), Int32(7))   => 42,
        (Int32(-3), Int32(5))  => -15,
    ] name = "mul_i32"

    @wasm_check sub_i32(Int32, Int32) [
        (Int32(10), Int32(3))  => 7,
    ] name = "sub_i32"
end

# Float ops verify that the wasm32 backend lowers Julia floats correctly
# — these go through different LLVM intrinsics from integer ops.
f32_double(x::Float32) = x * 2.0f0
f64_add(a::Float64, b::Float64) = a + b

@testset "M1 · floating-point" begin
    @wasm_check f32_double(Float32) [
        (Float32(1.5),)  => 3.0,
        (Float32(-2.25),) => -4.5,
    ] name = "f32_double"

    @wasm_check f64_add(Float64, Float64) [
        (1.5, 2.5)     => 4.0,
        (-1e10, 1e10)  => 0.0,
    ] name = "f64_add"
end

# Branches: tests that the wasm backend's structurization handles
# diamond control flow (the simplest case before loops).
abs_i32(x::Int32) = x < Int32(0) ? -x : x
sign_i32(x::Int32) = x > Int32(0) ? Int32(1) : (x < Int32(0) ? Int32(-1) : Int32(0))

@testset "M1 · simple branches" begin
    @wasm_check abs_i32(Int32) [
        (Int32(0),) => 0,
        (Int32(7),) => 7,
        (Int32(-7),) => 7,
        (Int32(typemin(Int32) + 1),) => -(typemin(Int32) + 1),
    ] name = "abs_i32"

    @wasm_check sign_i32(Int32) [
        (Int32(7),)  => 1,
        (Int32(-7),) => -1,
        (Int32(0),)  => 0,
    ] name = "sign_i32"
end

# Loops: gauss-style sum, while loop, no allocation. Validates the
# wasm relooper turns Julia's CFG into structured wasm `loop`/`block`.
function sum_to(n::Int32)::Int32
    s = Int32(0)
    i = Int32(1)
    while i <= n
        s += i
        i += Int32(1)
    end
    return s
end

@testset "M1 · while loop" begin
    @wasm_check sum_to(Int32) [
        (Int32(0),)  => 0,
        (Int32(1),)  => 1,
        (Int32(10),) => 55,
        (Int32(100),) => 5050,
    ] name = "sum_to"
end

# Recursion: confirms call lowering and stack frames work.
function fact_i32(n::Int32)::Int32
    n <= Int32(1) ? Int32(1) : n * fact_i32(n - Int32(1))
end

@testset "M1 · recursion" begin
    @wasm_check fact_i32(Int32) [
        (Int32(0),)  => 1,
        (Int32(1),)  => 1,
        (Int32(5),)  => 120,
        (Int32(10),) => 3628800,
    ] name = "fact_i32"
end

# Cross-function call: validates that GPUCompiler emits the callee
# alongside the entry point and lld keeps both reachable.
sq(x::Int32) = x * x
sum_of_squares(a::Int32, b::Int32) = sq(a) + sq(b)

@testset "M1 · cross-function call" begin
    @wasm_check sum_of_squares(Int32, Int32) [
        (Int32(3), Int32(4)) => 25,
        (Int32(0), Int32(0)) => 0,
    ] name = "sum_of_squares"
end
