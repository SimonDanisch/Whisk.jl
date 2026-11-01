## M12 — image upload + pin + whisk_gc_collect end-to-end (#47).
##
## Reproduces the SimiMaps "memory access out of bounds" trap from #47:
## host-build a struct with Dict + BitVector + Vector fields, upload to
## wasm, pin the root, call an entry that reads the image, then run
## `whisk_gc_collect()` and call again. With a layout-drifted encoder
## the marker walks past the image's pointer slots → trap. With our
## `pointer_offsets`-driven encoder + `assert_addr_fixups_match!` it
## should round-trip cleanly.

using Test
using Whisk

# Module-scope so the type is identifiable across @testset blocks and
# the wasm-side `unwrap_value(M12, addr)` resolves to a stable type.
struct M12
    ids::Vector{Int64}
    lookup::Dict{Int64, UInt32}
    mask::BitVector
end

# Top-level entry — closure-defined entry tripped a separate codegen
# path (task #46 was about this exact shape). Returns a deterministic
# digest of the image's contents so we can verify the marker didn't
# clobber anything across the collect.
@noinline function m12_summary(addr::UInt64)::Int64
    g = Whisk.unwrap_value(M12, addr)
    s::Int64 = Int64(length(g.ids))
    s += Int64(length(g.lookup)) * Int64(7)
    s += Int64(length(g.mask)) * Int64(13)
    s += Int64(get(g.lookup, Int64(100), UInt32(0))) * Int64(17)
    return s
end

# JS harness: upload the image, pin root, call entry, run gc_collect,
# call entry again, assert both calls return the expected digest.
function run_m12_image_gc(wasm_path::AbstractString,
                            image::Vector{UInt8},
                            tag_ids::Vector{Int},
                            entry::AbstractString,
                            expected::Int64)
    img_hex   = bytes2hex(image)
    tag_ids_js = "[" * join(string.(tag_ids), ", ") * "]"
    js = """
    import fs from 'node:fs';
    const wasmBytes = fs.readFileSync(process.argv[1]);
    const closureBytes = Buffer.from(process.argv[3], 'hex');
    let mem = null;
    const Nn = b => typeof b === 'bigint' ? Number(b) : b;
    const imports = { env: {
        js_console: (l, p, n) => {
            const m = new TextDecoder().decode(new Uint8Array(mem.buffer, Nn(p), Nn(n)));
            (l === 2 ? console.error : console.log).call(console, "[wasm] " + m);
        },
        memmove: (d, s, n) => { new Uint8Array(mem.buffer).copyWithin(Nn(d), Nn(s), Nn(s)+Nn(n)); return d; },
        memcpy:  (d, s, n) => { new Uint8Array(mem.buffer).copyWithin(Nn(d), Nn(s), Nn(s)+Nn(n)); return d; },
        memset:  (p, v, n) => { new Uint8Array(mem.buffer).fill(Number(v) & 0xff, Nn(p), Nn(p)+Nn(n)); return p; },
    }};
    const { instance } = await WebAssembly.instantiate(wasmBytes, imports);
    const exp = instance.exports;
    mem = exp.memory;

    // Manual image upload (mirrors WhiskInstance.uploadImage).
    const tagIds = $tag_ids_js;
    const layoutAddrs = tagIds.map(t => Number(exp.__whisk_layout_addr(BigInt(t))));
    const hdr = new DataView(closureBytes.buffer, closureBytes.byteOffset, 16);
    const image_size      = hdr.getUint32(0, true);
    const root_offset     = hdr.getUint32(4, true);
    const n_addr_fixups   = hdr.getUint32(8, true);
    const n_layout_fixups = hdr.getUint32(12, true);

    // Use whisk_gc_reserve (no outer GC header): the image's embedded
    // sub-object headers ARE the only headers in this range, so sweep
    // walks each one individually rather than seeing one big block.
    const base = Number(exp.whisk_gc_reserve(BigInt(image_size)));
    new Uint8Array(mem.buffer, base, image_size).set(closureBytes.subarray(16, 16 + image_size));
    const mDV = new DataView(mem.buffer);
    const af = new DataView(closureBytes.buffer, closureBytes.byteOffset + 16 + image_size, 4 * n_addr_fixups);
    for (let i = 0; i < n_addr_fixups; i++) {
        const off = af.getUint32(i * 4, true);
        mDV.setBigUint64(base + off, mDV.getBigUint64(base + off, true) + BigInt(base), true);
    }
    const lf = new DataView(closureBytes.buffer,
                             closureBytes.byteOffset + 16 + image_size + 4 * n_addr_fixups,
                             8 * n_layout_fixups);
    for (let i = 0; i < n_layout_fixups; i++) {
        const off = lf.getUint32(i * 8, true);
        const tag = lf.getUint32(i * 8 + 4, true);
        mDV.setBigUint64(base + off, BigInt(layoutAddrs[tag]), true);
    }

    const rootAddr = base + root_offset;
    exp.whisk_pin(BigInt(rootAddr));

    const r1 = exp[process.argv[2]](BigInt(rootAddr));
    exp.whisk_gc_collect();              // ← the #47 trigger
    const r2 = exp[process.argv[2]](BigInt(rootAddr));
    exp.whisk_gc_collect();
    const r3 = exp[process.argv[2]](BigInt(rootAddr));

    process.stdout.write(r1.toString() + " " + r2.toString() + " " + r3.toString());
    """
    raw = read(`$NODE --input-type=module -e $js -- $wasm_path $entry $img_hex`, String)
    parts = split(strip(raw), " ")
    @assert length(parts) == 3 "expected 3 results, got: $raw"
    return parse.(Int64, parts)
end

@testset "M12 · image upload + pin + gc_collect round-trip (#47)" begin
    g = M12(
        [Int64(1), Int64(2), Int64(3)],
        Dict(Int64(100) => UInt32(42), Int64(200) => UInt32(99)),
        BitVector([true, false, true, true, false, true, false]),
    )
    expected = Int64(3) + Int64(2) * Int64(7) + Int64(7) * Int64(13) +
               Int64(42) * Int64(17)

    blob, tags = Whisk.convert2wasm(g)
    @test !isempty(blob)
    @test !isempty(tags.keys)

    out = tempname() * ".wasm"
    res = Whisk.compile_to_wasm(m12_summary, Tuple{UInt64};
                                  out_path = out, layout_tags = tags,
                                  name = "m12_summary")
    @test isfile(res.wasm_path)

    tag_ids = [Int(i - 1) for i in 1:length(tags.keys)]
    r1, r2, r3 = run_m12_image_gc(res.wasm_path, blob, tag_ids,
                                    res.entry_name, expected)
    @test r1 == expected
    @test r2 == expected     # ← if #47's bug were live, this would trap
    @test r3 == expected
end
