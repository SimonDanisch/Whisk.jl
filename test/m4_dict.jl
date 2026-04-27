## M4 — `Dict` / `Set` / `BitVector`. Tests that real stdlib data
## structures with significant runtime machinery (hashing, rehash,
## memory allocations, growth) compile and run end-to-end.

using Test

function dict_set_get(x::Int32)::Int32
    d = Dict{UInt8, Float32}()
    d[0x01] = 1f0
    d[0x02] = 2f0
    d[0x03] = 3f0
    return unsafe_trunc(Int32, d[unsafe_trunc(UInt8, x)])
end

@testset "M4 · Dict{UInt8, Float32} set + get" begin
    @wasm_check dict_set_get(Int32) [
        (Int32(1),) => 1,
        (Int32(2),) => 2,
        (Int32(3),) => 3,
    ] name = "dict_set_get"
end

# Empty Dict's length is 0 — verifies the empty-Memory rewrite catches
# const-prop'd host-singleton accesses.
function dict_empty_len()::Int32
    d = Dict{UInt8, Float32}()
    return unsafe_trunc(Int32, length(d))
end

@testset "M4 · Dict empty length" begin
    @wasm_check dict_empty_len() [
        () => 0,
    ] name = "dict_empty_len"
end

# A larger Dict with many keys exercises rehash and growth.
function dict_growth(n::Int32)::Int32
    d = Dict{UInt8, Float32}()
    i = Int32(1)
    while i <= n
        d[unsafe_trunc(UInt8, i)] = Float32(i)
        i += Int32(1)
    end
    return unsafe_trunc(Int32, length(d))
end

@testset "M4 · Dict{UInt8, Float32} grow + rehash" begin
    @wasm_check dict_growth(Int32) [
        (Int32(1),)  => 1,
        (Int32(8),)  => 8,
        (Int32(50),) => 50,
    ] name = "dict_growth"
end
