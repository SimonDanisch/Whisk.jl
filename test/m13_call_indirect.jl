## M13 — wasm function-table / call_indirect smoke test.
##
## This is intentionally a compiler-generated probe, not a Julia user
## feature yet. It proves that Whisk can emit an indirect function call,
## link it as wasm, export the table, and execute it under Node before
## we generalise the machinery to MethodInstance vtables.

using Test

function call_indirect_anchor(x::Int32)::Int32
    return x
end

function wasm_exports(wasm_path::AbstractString)
    js = """
    import fs from 'node:fs';
    const bytes = fs.readFileSync(process.argv[1]);
    const mod = new WebAssembly.Module(bytes);
    process.stdout.write(JSON.stringify(WebAssembly.Module.exports(mod)));
    """
    raw = read(`$NODE --input-type=module -e $js -- $wasm_path`, String)
    return raw
end

@testset "M13 · call_indirect probe" begin
    out = tempname() * ".wasm"
    old_strict = get(ENV, "WHISK_VALIDATE_STRICT", nothing)
    ENV["WHISK_VALIDATE_STRICT"] = "1"
    res = try
        Whisk.compile_to_wasm(call_indirect_anchor, Tuple{Int32};
                              out_path = out,
                              name = "call_indirect_anchor",
                              optimize = true,
                              emit_indirect_probe = true)
    finally
        old_strict === nothing ? delete!(ENV, "WHISK_VALIDATE_STRICT") :
                                 (ENV["WHISK_VALIDATE_STRICT"] = old_strict)
    end

    @test occursin("call i32 %", res.llvm_ir)
    @test occursin("\"kind\":\"table\"", wasm_exports(res.wasm_path))
    @test run_export(res.wasm_path, "whisk_call_indirect_probe", (Int32(41),);
                     rettype = Int32) == 42
end
