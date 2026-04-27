## M2+ — anything that allocates. Currently expected-broken; flips to
## green when the bump allocator + linear-memory runtime lands.
##
## Each broken test tries to actually RUN the wasm (not just compile it),
## because Julia's optimizer is aggressive — `length("hello")`
## constant-folds to a literal `5` and wouldn't catch missing allocator
## machinery. We force the allocation onto the runtime path with a
## non-constant size.

using Test

# Heap allocation that survives constant folding (size depends on input).
function alloc_and_first(n::Int32)::Int32
    v = Vector{Int32}(undef, Int(n))
    @inbounds v[1] = Int32(42)
    return @inbounds v[1]
end

@testset "M2 · Vector{Int32}(undef, n)" begin
    @wasm_check alloc_and_first(Int32) [
        (Int32(8),)  => 42,
        (Int32(64),) => 42,
    ] name = "alloc_and_first"
end

# Same shape for Float32 — separate codepath through Memory{Float32}.
function alloc_and_set(n::Int32)::Float32
    v = Vector{Float32}(undef, Int(n))
    @inbounds v[1] = 3.14f0
    return @inbounds v[1]
end

@testset "M2 · Vector{Float32}(undef, n)" begin
    @wasm_check alloc_and_set(Int32) [
        (Int32(4),)  => Float32(3.14),
        (Int32(32),) => Float32(3.14),
    ] name = "alloc_and_set"
end

# String operation that can't constant-fold (string from runtime input).
# Forces String overlay machinery onto the path.
function string_byte(i::Int32)::UInt8
    s = "hello"
    @inbounds codeunit(s, Int(i))
end

@testset "M3/M4 · String codeunit (broken until later)" begin
    @test_broken try
        res = compile_module(string_byte, Tuple{Int32}; name = "string_byte")
        run_export(res.wasm_path, res.entry_name, (Int32(1),)) == Int(UInt8('h'))
    catch
        false
    end
end
