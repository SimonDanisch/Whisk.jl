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

    // The wasm module ships its own bump allocator (M3) — no
    // host-side malloc. JS only provides error stubs and the C-library
    // bulk-memory functions that LLVM may emit references to.
    let mem = null;
    // wasm64 passes 64-bit integer args/results as JS BigInts.
    // Convert at the boundary: BigInt → Number for typed-array
    // offsets / lengths (heap fits in 53 bits, no precision loss),
    // and Number → BigInt for return values.
    const N = (b) => typeof b === 'bigint' ? Number(b) : b;
    const imports = {
      env: {
        jl_argument_error(_p) { throw new Error("argument error"); },
        gpu_report_exception(_p) { throw new Error("gpu_report_exception"); },
        gpu_report_oom(_sz) { throw new Error("gpu_report_oom"); },
        gpu_signal_exception() { throw new Error("gpu_signal_exception"); },
        // js_console(level, ptr, len) — Whisk's exception/log channel.
        // level: 0=log, 1=warn, 2=error. UTF-8 bytes from linear memory.
        js_console(level, ptr, len) {
          const pn = N(ptr), ln = N(len);
          const bytes = new Uint8Array(mem.buffer, pn, ln);
          const msg = new TextDecoder('utf-8').decode(bytes);
          const fn = level === 2 ? console.error
                   : level === 1 ? console.warn
                                 : console.log;
          fn.call(console, "[wasm] " + msg);
        },
        memset(p, val, n) {
          const pn = N(p), nn = N(n);
          new Uint8Array(mem.buffer).fill(Number(val) & 0xff, pn, pn + nn);
          return p;
        },
        memcpy(dst, src, n) {
          const dn = N(dst), sn = N(src), nn = N(n);
          new Uint8Array(mem.buffer).copyWithin(dn, sn, sn + nn);
          return dst;
        },
        memmove(dst, src, n) {
          const dn = N(dst), sn = N(src), nn = N(n);
          new Uint8Array(mem.buffer).copyWithin(dn, sn, sn + nn);
          return dst;
        },
      },
    };

    const { instance } = await WebAssembly.instantiate(bytes, imports);
    mem = instance.exports.memory;
    const r = instance.exports[process.argv[2]]($args_js);
    process.stdout.write(String(r));
    """
    raw = read(`$NODE --input-type=module -e $js -- $wasm_path $entry`, String)
    return _parse_node_result(raw, rettype)
end

"""
    run_blob_export(wasm_path, entry, blob; rettype = Int32) → value

Place `blob` (a `Vector{UInt8}`) into wasm linear memory at a fixed
high offset, call `entry(blob_ptr, blob_len)`, and parse the printed
return as `rettype`. Used for `Whisk.Marshal` end-to-end tests where
the wasm module needs raw bytes to decode.
"""
function run_blob_export(wasm_path::AbstractString, entry::AbstractString,
                          blob::Vector{UInt8}; rettype::Type = Int32)
    blob_hex = bytes2hex(blob)
    js = """
    import fs from 'node:fs';
    const wasmBytes = fs.readFileSync(process.argv[1]);
    const blob = Buffer.from(process.argv[3], 'hex');
    let mem = null;
    const N = (b) => typeof b === 'bigint' ? Number(b) : b;
    const imports = { env: {
        js_console(level, ptr, len) {
            const m = new TextDecoder().decode(new Uint8Array(mem.buffer, N(ptr), N(len)));
            (level === 2 ? console.error : console.log).call(console, "[wasm] " + m);
        },
        memmove(d, s, n) { new Uint8Array(mem.buffer).copyWithin(N(d), N(s), N(s) + N(n)); return d; },
        memset(p, v, n) { new Uint8Array(mem.buffer).fill(Number(v) & 0xff, N(p), N(p) + N(n)); return p; },
    }};
    const { instance } = await WebAssembly.instantiate(wasmBytes, imports);
    mem = instance.exports.memory;
    const off = 8 * 1024 * 1024;  // well past the wasm runtime's bump heap
    new Uint8Array(mem.buffer).set(blob, off);
    const r = instance.exports[process.argv[2]](BigInt(off), BigInt(blob.length));
    process.stdout.write(String(r));
    """
    raw = read(`$NODE --input-type=module -e $js -- $wasm_path $entry $blob_hex`, String)
    return parse(rettype, strip(raw))
end

"""
    inspect_gc(wasm_path) → NamedTuple

Run a wasm module under JS that interleaves its main `call` with
`whisk_gc_collect()` invocations, and reports `_whisk_heap_next` /
`_whisk_free_list_head` along the way. Used for GC functional tests.

The wasm must export `whisk_gc_collect` (Whisk does this at link time).
"""
function gc_probe(wasm_path::AbstractString, entry::AbstractString, args; rettype = nothing)
    args_js = join(string.(args), ", ")
    js = """
    import fs from 'node:fs';
    const bytes = fs.readFileSync(process.argv[1]);
    let mem = null;
    const N = (b) => typeof b === 'bigint' ? Number(b) : b;
    const imports = { env: {
        jl_argument_error(_p) { throw new Error("argument error"); },
        gpu_report_exception(_p) { throw new Error("gpu_report_exception"); },
        gpu_report_oom(_sz) { throw new Error("gpu_report_oom"); },
        gpu_signal_exception() { throw new Error("gpu_signal_exception"); },
        js_console(level, ptr, len) {
          const pn = N(ptr), ln = N(len);
          const bytes = new Uint8Array(mem.buffer, pn, ln);
          const msg = new TextDecoder('utf-8').decode(bytes);
          (level === 2 ? console.error : level === 1 ? console.warn : console.log)
            .call(console, "[wasm] " + msg);
        },
        memset(p, val, n) {
          new Uint8Array(mem.buffer).fill(val & 0xff, p, p + n); return p;
        },
        memcpy(dst, src, n) {
          new Uint8Array(mem.buffer).copyWithin(dst, src, src + n); return dst;
        },
        memmove(dst, src, n) {
          new Uint8Array(mem.buffer).copyWithin(dst, src, src + n); return dst;
        },
    }};
    const { instance } = await WebAssembly.instantiate(bytes, imports);
    mem = instance.exports.memory;

    // Run the entry, then explicitly invoke whisk_gc_collect, then
    // re-run a second time to verify the GC didn't break anything.
    const r1 = instance.exports[process.argv[2]]($args_js);
    instance.exports.whisk_gc_collect();
    const r2 = instance.exports[process.argv[2]]($args_js);
    instance.exports.whisk_gc_collect();
    process.stdout.write(JSON.stringify({ r1: Number(r1), r2: Number(r2) }));
    """
    raw = read(`$NODE --input-type=module -e $js -- $wasm_path $entry`, String)
    return raw
end

"""
    run_closure_export(wasm_path, closure_image, manifest, entry, args; rettype = nothing)

Drive a wasm module that was compiled from a closure (callable struct
with captured fields) — the path that auto-uploads `closure_image`,
calls `whisk_set_closure_root`, and only then invokes `entry(args...)`.

Loads `whisk.js` from the package source so the test exercises the
real public JS API (not a duplicate of its internals).
"""
function run_closure_export(wasm_path::AbstractString,
                              closure_image::Vector{UInt8},
                              manifest,
                              entry::AbstractString, args;
                              rettype::Union{Type, Nothing} = nothing)
    # wasm64 i64 args need JS BigInt literals; floats stay as Number.
    args_js = join((a isa Integer ? string(Int(a)) * "n" : string(a)
                    for a in args), ", ")
    # whisk.js's `_resolveLayouts` only reads `manifest.tags[i].tag` to
    # call `__whisk_layout_addr` on. We're generating JS source, not
    # serialising for a wire — splat the tag IDs straight into a JS
    # object literal.
    tag_ids_js = manifest === nothing ? "[]" :
                 "[" * join((string(t.tag) for t in manifest.tags), ", ") * "]"
    img_hex = bytes2hex(closure_image)
    whisk_js_url = "file://" * abspath(joinpath(@__DIR__, "..", "src",
                                                  "jsinterop", "whisk.js"))
    js = """
    import fs from 'node:fs';
    import { instantiate } from $(repr(whisk_js_url));
    const wasmBytes = fs.readFileSync(process.argv[1]);
    const closureBytes = Buffer.from(process.argv[3], 'hex');
    const manifest = { tags: $tag_ids_js.map(t => ({ tag: t })) };
    const wi = await instantiate(wasmBytes, {
        manifest,
        closureImage: closureBytes,
    });
    const r = wi.exports[process.argv[2]]($args_js);
    process.stdout.write(String(r));
    """
    raw = read(`$NODE --input-type=module -e $js -- $wasm_path $entry $img_hex`, String)
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
