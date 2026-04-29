## M5 — math intrinsics. Trig + sqrt + asin compose into haversine, the
## first real piece of routing-shaped code Whisk handles end-to-end.
##
## All Julia's math goes through the regular GPUCompiler pipeline; the
## only special handling is `link_compiler_rt!` filling in the i128
## helpers (`__multi3` / `__lshrti3` / `__ashlti3`) that Float64 trig
## argument-reduction pulls in via Int128 internals.

using Test

# Pure scalar trig — exercises sin / cos chain for moderate inputs.
function trig_chain(x::Float64)::Float64
    sin(x) * cos(x) + sin(2 * x) / 2
end

@testset "M5 · sin/cos basic chain" begin
    @wasm_check trig_chain(Float64) [
        (0.0,)       => 0.0,
        (1.0,)       => trig_chain(1.0),
        (3.14159,)   => trig_chain(3.14159),
    ] name = "trig_chain" rettype = Float64
end

# sqrt → wasm `f64.sqrt` opcode; no libcalls.
function hypot2(a::Float64, b::Float64)::Float64
    sqrt(a*a + b*b)
end

@testset "M5 · sqrt + Float64 arithmetic" begin
    @wasm_check hypot2(Float64, Float64) [
        (3.0, 4.0)   => 5.0,
        (5.0, 12.0)  => 13.0,
        (0.0, 0.0)   => 0.0,
    ] name = "hypot2" rettype = Float64
end

# asin — closed-form, separate codepath from sin/cos. Range [-1,1].
function asin_clamped(x::Float64)::Float64
    asin(min(1.0, max(-1.0, x)))
end

@testset "M5 · asin with clamp" begin
    @wasm_check asin_clamped(Float64) [
        (0.0,)   => 0.0,
        (1.0,)   => asin(1.0),
        (-1.0,)  => asin(-1.0),
        (0.5,)   => asin(0.5),
    ] name = "asin_clamped" rettype = Float64
end

# Real haversine — first chunk of SimiMaps' routing/astar.jl that runs
# unmodified under wasm. Combines deg2rad, sin, cos, sqrt, asin, min,
# Float32 conversion, and i128-via-trig-argument-reduction.
function haversine_m(lat1::Float64, lon1::Float64, lat2::Float64, lon2::Float64)::Float32
    R = 6_371_000.0
    φ1 = deg2rad(lat1); φ2 = deg2rad(lat2)
    Δφ = deg2rad(lat2 - lat1); Δλ = deg2rad(lon2 - lon1)
    a = sin(Δφ/2)^2 + cos(φ1)*cos(φ2)*sin(Δλ/2)^2
    return Float32(2 * R * asin(min(1.0, sqrt(a))))
end

@testset "M5 · haversine_m (real SimiMaps routing primitive)" begin
    @wasm_check haversine_m(Float64, Float64, Float64, Float64) [
        (52.5, 13.4, 52.6, 13.5)              => haversine_m(52.5, 13.4, 52.6, 13.5),
        (0.0, 0.0, 0.0, 1.0)                  => haversine_m(0.0, 0.0, 0.0, 1.0),
        (52.520008, 13.404954, 48.8566, 2.3522) => haversine_m(52.520008, 13.404954, 48.8566, 2.3522),
        (0.0, 0.0, 90.0, 0.0)                 => haversine_m(0.0, 0.0, 90.0, 0.0),
    ] name = "haversine_m" rettype = Float32
end
