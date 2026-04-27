## Test helpers — compile a Julia function, run the resulting wasm under
## Node, return the result as a parsed Julia value. Single source of
## truth for "does this Julia code work end-to-end" tests.

using Whisk
using Test

const NODE = Sys.which("node")
NODE === nothing &&
    error("Whisk tests require `node` on PATH (used to run wasm under V8)")

"""
    compile_module(f, argtypes; name = nothing) → NamedTuple

Thin wrapper that picks a temp path and asserts the wasm file actually
got created. Returns whatever `Whisk.compile_to_wasm` returns.
"""
function compile_module(@nospecialize(f), @nospecialize(argtypes::Type{<:Tuple});
                        name::Union{Nothing,String} = nothing)
    out = tempname() * ".wasm"
    res = Whisk.compile_to_wasm(f, argtypes;
                                out_path = out,
                                name     = name,
                                optimize = true)
    @assert isfile(res.wasm_path) "wasm file missing at $(res.wasm_path)"
    return res
end

"""
    run_export(wasm_path, entry, args; rettype = nothing) → value

Spawn Node once, instantiate the wasm module, call `entry(args...)`,
parse the printed result as `rettype` (defaults to inferring from the
output text — Int / Float64 / String).
"""
function run_export(wasm_path::AbstractString, entry::AbstractString, args;
                    rettype::Union{Type, Nothing} = nothing)
    # `string(::Int32) → "2"`, `string(Float32(1.5)) → "1.5"` — these
    # parse as JS literals. `repr` would emit `"Int32(2)"` which doesn't.
    args_js = join(string.(args), ", ")
    js = """
    import fs from 'node:fs';
    const bytes = fs.readFileSync(process.argv[1]);
    const { instance } = await WebAssembly.instantiate(bytes, {});
    const r = instance.exports[process.argv[2]]($args_js);
    process.stdout.write(String(r));
    """
    raw = read(`$NODE --input-type=module -e $js -- $wasm_path $entry`, String)
    return _parse_node_result(raw, rettype)
end

# ── Map JS's `String(x)` output back to a Julia value. We default to
# Float64 if there's a decimal point, Int otherwise, and fall back to
# the raw string. Callers can pin the type via `rettype = Float32` etc.
function _parse_node_result(raw::AbstractString, rettype)
    s = strip(raw)
    rettype === nothing || return parse(rettype, s)
    occursin('.', s) || occursin('e', s) ?
        parse(Float64, s) :
        parse(Int, s)
end

"""
    @wasm_check f(argtypes...) cases [name=...]

Compile `f` and assert each `(input...) => expected` case runs in V8
and returns the expected value. A single helper so each test is one
line and the regression catalogue stays readable.
"""
macro wasm_check(call_expr, cases_expr, kwargs...)
    @assert call_expr.head === :call "expected `f(T1, T2, ...)`"
    fname  = call_expr.args[1]
    argTs  = Expr(:curly, :Tuple, call_expr.args[2:end]...)
    name_kw = filter(kw -> kw.args[1] === :name, collect(kwargs))
    rt_kw   = filter(kw -> kw.args[1] === :rettype, collect(kwargs))
    namearg = isempty(name_kw) ? :nothing : esc(name_kw[1].args[2])
    rtarg   = isempty(rt_kw)   ? :nothing : esc(rt_kw[1].args[2])
    quote
        let res = compile_module($(esc(fname)), $(esc(argTs)); name = $namearg)
            entry = res.entry_name
            for (input, expected) in $(esc(cases_expr))
                got = run_export(res.wasm_path, entry, input; rettype = $rtarg)
                @test got == expected
            end
            res  # return module info for follow-up assertions
        end
    end
end
