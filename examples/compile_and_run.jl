## Compile a Julia function to WebAssembly and run it with Node's
## WebAssembly runtime.
##
## Usage:
##     julia --project=. examples/compile_and_run.jl

using Whisk

function sum_squares(n::Int32)::Int32
    total = Int32(0)
    for i in Int32(1):n
        total += i * i
    end
    return total
end

function run_wasm_export(wasm_path::AbstractString, entry::AbstractString,
                         args::Tuple)
    node = Sys.which("node")
    node === nothing && error("Node.js is required to run this example")

    args_js = join(string.(args), ", ")
    js = """
    import fs from 'node:fs';

    const bytes = fs.readFileSync(process.argv[1]);
    let mem = null;
    const N = (x) => typeof x === 'bigint' ? Number(x) : x;

    const imports = {
      env: {
        jl_argument_error(_p) { throw new Error("argument error"); },
        gpu_report_exception(_p) { throw new Error("gpu_report_exception"); },
        gpu_report_oom(_sz) { throw new Error("gpu_report_oom"); },
        gpu_signal_exception() { throw new Error("gpu_signal_exception"); },
        js_console(level, ptr, len) {
          const msg = new TextDecoder("utf-8")
            .decode(new Uint8Array(mem.buffer, N(ptr), N(len)));
          const log = level === 2 ? console.error
                    : level === 1 ? console.warn
                                  : console.log;
          log.call(console, "[wasm] " + msg);
        },
        memset(p, value, n) {
          new Uint8Array(mem.buffer).fill(Number(value) & 0xff, N(p), N(p) + N(n));
          return p;
        },
        memcpy(dst, src, n) {
          new Uint8Array(mem.buffer).copyWithin(N(dst), N(src), N(src) + N(n));
          return dst;
        },
        memmove(dst, src, n) {
          new Uint8Array(mem.buffer).copyWithin(N(dst), N(src), N(src) + N(n));
          return dst;
        },
      },
    };

    const { instance } = await WebAssembly.instantiate(bytes, imports);
    mem = instance.exports.memory;

    const result = instance.exports[process.argv[2]]($args_js);
    process.stdout.write(String(result));
    """

    raw = read(`$node --input-type=module -e $js -- $wasm_path $entry`, String)
    return parse(Int32, strip(raw))
end

out_path = joinpath(mktempdir(), "sum_squares.wasm")
compiled = Whisk.compile_to_wasm(sum_squares, Tuple{Int32};
                                 out_path,
                                 name = "sum_squares",
                                 optimize = true)

input = Int32(10)
got = run_wasm_export(compiled.wasm_path, compiled.entry_name, (input,))
expected = sum_squares(input)

println("wasm  : ", compiled.wasm_path, " (", filesize(compiled.wasm_path), " bytes)")
println("entry : ", compiled.entry_name)
println("call  : ", compiled.entry_name, "(", input, ")")
println("result: ", got)

got == expected || error("expected $expected, got $got")
