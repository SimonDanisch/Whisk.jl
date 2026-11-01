// Whisk JS-side companion library — single ES module that handles every
// piece of plumbing for calling Whisk-compiled Julia wasm from JavaScript.
//
// Goals (mirroring wasm-bindgen for Rust):
//   * One-call instantiation: `await Whisk.instantiate(wasmBytes, opts)`
//     auto-satisfies the env imports Whisk emits (memset/memcpy/memmove/
//     js_console), wraps allocator + pin/unpin/collect, and exposes the
//     wasm exports plus a uniform conversion surface.
//   * `js2wasm`: wasm-bindgen-style cross-boundary helpers for the
//     classic JS shapes — primitives pass through, strings/bytes/typed-
//     arrays upload to wasm linear memory and return `(ptr, len)` pairs
//     ready to splat into a wasm export call.
//   * `uploadImage`: takes a host-built memory image (produced by
//     `Whisk.convert2wasm` Julia-side) and lands it in wasm memory with
//     pointer + layout-descriptor fixups, returning a FinalizationRegistry-
//     backed `WhiskHandle` whose lifetime drives `whisk_unpin`.
//
// No external deps. Distribute as a single `.mjs` file, served alongside
// the wasm. Bonito-friendly: import via `import { Whisk } from "..."`.
//
// Image format (matches Whisk.convert2wasm output, all little-endian):
//
//     u32 image_size
//     u32 root_offset           // objref of root inside image
//     u32 n_addr_fixups
//     u32 n_layout_fixups
//     u8  data[image_size]
//     u32 addr_fixups[n_addr_fixups]              // += base after memcpy
//     {u32 offset, u32 type_tag}[n_layout_fixups] // GC-header layout slot

const TEXT_ENC = new TextEncoder();
const TEXT_DEC = new TextDecoder('utf-8');

// ── Marshal: wire-format reader matching Whisk.Marshal ──────────────
//
// Whisk-compiled wasm functions that return complex Julia values usually
// encode them via `Whisk.Marshal.encode(io, x)` into a caller-provided
// byte buffer. This reader is the JS-side mirror of `Whisk.Marshal`'s
// ByteReader. It's wire-compatible: read the same primitives in the
// same order Julia wrote them and you get the same value tree.
//
// Usage: pass an ArrayBuffer (or Uint8Array.buffer) and read field-by-
// field using the type tags defined for your output struct.

class ByteReader {
    constructor(buf) {
        // Accept ArrayBuffer, Uint8Array, or DataView.
        if (buf instanceof Uint8Array) {
            this.view = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
        } else if (buf instanceof DataView) {
            this.view = buf;
        } else {
            this.view = new DataView(buf);
        }
        this.pos = 0;
    }
    u8()   { const v = this.view.getUint8(this.pos);             this.pos += 1; return v; }
    i8()   { const v = this.view.getInt8(this.pos);              this.pos += 1; return v; }
    u16()  { const v = this.view.getUint16(this.pos, true);      this.pos += 2; return v; }
    i16()  { const v = this.view.getInt16(this.pos, true);       this.pos += 2; return v; }
    u32()  { const v = this.view.getUint32(this.pos, true);      this.pos += 4; return v; }
    i32()  { const v = this.view.getInt32(this.pos, true);       this.pos += 4; return v; }
    u64()  { const v = this.view.getBigUint64(this.pos, true);   this.pos += 8; return Number(v); }
    i64()  { const v = this.view.getBigInt64(this.pos, true);    this.pos += 8; return Number(v); }
    u64big() { const v = this.view.getBigUint64(this.pos, true); this.pos += 8; return v; }
    i64big() { const v = this.view.getBigInt64(this.pos, true);  this.pos += 8; return v; }
    f32()  { const v = this.view.getFloat32(this.pos, true);     this.pos += 4; return v; }
    f64()  { const v = this.view.getFloat64(this.pos, true);     this.pos += 8; return v; }
    bool() { return this.u8() !== 0; }
    /** Read a u32 length, then `n` elements via `elem(reader)`. */
    vector(elem) {
        const n = this.u32(); const out = new Array(n);
        for (let i = 0; i < n; i++) out[i] = elem(this);
        return out;
    }
    /** Read a u32 length and that many UTF-8 bytes as a string. */
    string() {
        const n = this.u32();
        const b = new Uint8Array(this.view.buffer, this.view.byteOffset + this.pos, n);
        this.pos += n;
        return TEXT_DEC.decode(b);
    }
}

// ── Handle wrapper ───────────────────────────────────────────────────
//
// Wraps a pinned wasm address in a JS object whose lifetime drives an
// automatic `whisk_unpin` on V8 GC. Manual `release()` is also fine —
// after release the registry callback is a no-op (we unregister).

class WhiskHandle {
    constructor(addr, registry, exportsRef) {
        // `addr` is a u64 wasm pointer; we keep it as BigInt so callers
        // can splat it straight into `wasm_export(handle.addr, ...)`.
        this.addr = typeof addr === 'bigint' ? addr : BigInt(addr);
        this._reg = registry;
        this._exp = exportsRef;
        this._released = false;
        Object.freeze(this);
    }
    release() {
        if (this._released) return;
        // `_released` is non-writable on the frozen object; we can't
        // mutate it. Use the registry's unregister token instead so the
        // GC callback short-circuits, and call unpin once.
        this._reg.unregister(this);
        try { this._exp.value.whisk_unpin(this.addr); }
        catch (e) { console.warn('WhiskHandle.release: unpin failed', e); }
    }
}

function _makeHandleFactory(exportsRef) {
    const reg = new FinalizationRegistry(addr => {
        try { exportsRef.value.whisk_unpin(BigInt(addr)); }
        catch (e) { /* wasm instance torn down — ignore */ }
    });
    return (addr) => {
        const h = new WhiskHandle(addr, reg, exportsRef);
        // Pass `h` itself as the unregister token so manual release()
        // can cancel the auto-unpin.
        reg.register(h, Number(h.addr), h);
        return h;
    };
}

// ── Default env imports ──────────────────────────────────────────────
//
// Whisk-compiled wasm currently imports `memset`/`memcpy`/`memmove`
// (libc fallbacks the wasm backend emits without bulk-memory enabled)
// and `js_console` (Whisk's diagnostic stream). Provide JS impls that
// cover every wasm fn import on the module — anything caller-supplied
// in `opts.imports.env` overrides these defaults.

function _buildEnv(module, instanceRef, callerEnv = {}) {
    const env = {};
    const memOf = () => instanceRef.value && instanceRef.value.exports.memory;
    const N = b => typeof b === 'bigint' ? Number(b) : b;

    const defaults = {
        memset: (p, v, n) => {
            const m = memOf(); if (!m) return p;
            new Uint8Array(m.buffer).fill(Number(v) & 0xff, N(p), N(p) + N(n));
            return p;
        },
        memcpy: (d, s, n) => {
            const m = memOf(); if (!m) return d;
            const dn = N(d), sn = N(s), ln = N(n);
            new Uint8Array(m.buffer, dn, ln)
                .set(new Uint8Array(m.buffer, sn, ln));
            return d;
        },
        memmove: (d, s, n) => {
            const m = memOf(); if (!m) return d;
            new Uint8Array(m.buffer).copyWithin(N(d), N(s), N(s) + N(n));
            return d;
        },
        js_console: (level, ptr, len) => {
            const m = memOf(); if (!m) return;
            const txt = TEXT_DEC.decode(new Uint8Array(m.buffer, N(ptr), N(len)));
            (level === 2 ? console.error :
             level === 1 ? console.warn  : console.log).call(console, '[wasm] ' + txt);
        },
    };

    for (const desc of WebAssembly.Module.imports(module)) {
        if (desc.module !== 'env' || desc.kind !== 'function') continue;
        env[desc.name] = (callerEnv[desc.name] !== undefined)
            ? callerEnv[desc.name]
            : (defaults[desc.name] !== undefined
                ? defaults[desc.name]
                : (...args) => {
                    throw new Error(`whisk: unstubbed wasm import env.${desc.name}(${args.length} args)`);
                });
    }
    return env;
}

// ── Main instance ───────────────────────────────────────────────────

class WhiskInstance {
    constructor(instance, manifest) {
        this._instance = instance;
        this.exports   = instance.exports;
        this.memory    = instance.exports.memory;
        this.manifest  = manifest || null;

        // Closure-scoped reference so handles can find the instance even
        // if a future `restart()` swaps it.
        this._exportsRef = { value: instance.exports };
        this._handle = _makeHandleFactory(this._exportsRef);

        // Cache resolved layout-descriptor addresses (one switch-call per
        // tag is fine, but uploads do them in a tight loop — cache once).
        this._layoutAddrs = null;
    }

    // ── Allocator + GC primitives ───────────────────────────────────

    /** Allocate `n` bytes of raw wasm-heap memory (layout=0). Returns BigInt. */
    malloc(n) { return this.exports.whisk_gc_alloc(BigInt(n)); }

    /** Pin a wasm address so mark+sweep doesn't reclaim it. */
    pin(addr)   { this.exports.whisk_pin(typeof addr === 'bigint' ? addr : BigInt(addr)); }
    unpin(addr) { this.exports.whisk_unpin(typeof addr === 'bigint' ? addr : BigInt(addr)); }

    /** Run a full mark+sweep at this safepoint (between wasm calls). */
    collect() { this.exports.whisk_gc_collect(); }

    /** Wrap a wasm address in a FinalizationRegistry-backed handle. */
    handle(addr) { return this._handle(addr); }

    // ── Memory views (re-derive on every access; mem.buffer detaches on grow) ──

    get heapU8() { return new Uint8Array(this.memory.buffer); }
    get heapDV() { return new DataView(this.memory.buffer); }

    // ── Layout descriptor lookup (manifest-driven) ──────────────────

    /** Resolve the wasm-rodata address of the LayoutDescriptor for `tag`. */
    layoutAddrOf(tag) {
        if (!this.exports.__whisk_layout_addr) {
            throw new Error('whisk: wasm has no __whisk_layout_addr export — was layout_tags passed to compile_to_wasm?');
        }
        return Number(this.exports.__whisk_layout_addr(BigInt(tag)));
    }

    /** Pre-cache all layout addresses listed in the manifest. */
    _resolveLayouts() {
        if (this._layoutAddrs) return this._layoutAddrs;
        if (!this.manifest || !this.manifest.tags) {
            throw new Error('whisk: instance has no manifest; pass {manifest} to instantiate()');
        }
        this._layoutAddrs = this.manifest.tags.map(t => this.layoutAddrOf(t.tag));
        return this._layoutAddrs;
    }

    // ── js2wasm: wasm-bindgen-style cross-boundary conversions ──────
    //
    // Primitives are passed verbatim by the caller (wasm i32/i64/f32/f64
    // map to JS Number/BigInt). Buffers and strings need a JS-side write
    // into wasm memory — these helpers do that uniformly. All of them
    // return `{ ptr, len }` ready to splat into a wasm function call.
    //
    // Wasm-side decoders consume `(ptr, len)`. For owned values, the
    // wasm function `whisk_pin`s the buffer it allocates from the bytes;
    // for borrowed reads it just reads + returns. Caller chooses.

    /** Upload UTF-8 bytes of `s` into wasm memory. Returns `{ ptr: bigint, len: number }`. */
    uploadString(s) {
        const bytes = TEXT_ENC.encode(s);
        return this.uploadBytes(bytes);
    }

    /** Upload raw bytes (Uint8Array or array-like) into wasm memory. */
    uploadBytes(u8) {
        if (!(u8 instanceof Uint8Array)) u8 = new Uint8Array(u8);
        const ptr = this.malloc(u8.byteLength);
        new Uint8Array(this.memory.buffer, Number(ptr), u8.byteLength).set(u8);
        return { ptr, len: u8.byteLength };
    }

    /** Upload a Float32Array (or convertible). Returns `{ ptr, len }`, len in elements. */
    uploadF32Array(arr) {
        const ta = arr instanceof Float32Array ? arr : new Float32Array(arr);
        const u8 = new Uint8Array(ta.buffer, ta.byteOffset, ta.byteLength);
        const { ptr } = this.uploadBytes(u8);
        return { ptr, len: ta.length };
    }

    /** Upload a Float64Array (or convertible). Returns `{ ptr, len }`, len in elements. */
    uploadF64Array(arr) {
        const ta = arr instanceof Float64Array ? arr : new Float64Array(arr);
        const u8 = new Uint8Array(ta.buffer, ta.byteOffset, ta.byteLength);
        const { ptr } = this.uploadBytes(u8);
        return { ptr, len: ta.length };
    }

    /** Upload an Int32Array (or convertible). Returns `{ ptr, len }`, len in elements. */
    uploadI32Array(arr) {
        const ta = arr instanceof Int32Array ? arr : new Int32Array(arr);
        const u8 = new Uint8Array(ta.buffer, ta.byteOffset, ta.byteLength);
        const { ptr } = this.uploadBytes(u8);
        return { ptr, len: ta.length };
    }

    /** Upload a BigInt64Array (or convertible). Returns `{ ptr, len }`, len in elements. */
    uploadI64Array(arr) {
        const ta = arr instanceof BigInt64Array ? arr : BigInt64Array.from(arr, BigInt);
        const u8 = new Uint8Array(ta.buffer, ta.byteOffset, ta.byteLength);
        const { ptr } = this.uploadBytes(u8);
        return { ptr, len: ta.length };
    }

    // ── Image upload: complex Julia values built host-side ──────────
    //
    // The host has called `Whisk.convert2wasm(value)` to produce
    // `(blob, tags)`. This method takes that blob, lays it into wasm
    // linear memory, applies pointer + layout-descriptor fixups, pins
    // the root, and returns a handle. The handle's `addr` is what wasm
    // exports expect when they do `Whisk.unwrap_value(T, addr)`.

    /**
     * Upload a host-built memory image and pin its root.
     * @param {Uint8Array|ArrayBuffer} blob — the image bytes
     * @returns {WhiskHandle} — auto-unpinned on JS GC; explicit `.release()` also fine
     */
    uploadImage(blob) {
        const u8 = blob instanceof Uint8Array ? blob : new Uint8Array(blob);
        const hdr = new DataView(u8.buffer, u8.byteOffset, 16);
        const image_size      = hdr.getUint32(0,  true);
        const root_offset     = hdr.getUint32(4,  true);
        const n_addr_fixups   = hdr.getUint32(8,  true);
        const n_layout_fixups = hdr.getUint32(12, true);

        const layoutAddrs = this._resolveLayouts();

        // Reserve raw bytes WITHOUT a wrapping GC header. The image's own
        // embedded sub-object headers are then the only headers in this
        // range — sweep walks each individually, marks/sweeps work
        // per-object. Going through `whisk_gc_alloc` would put the whole
        // image inside one outer block whose mark bit is the only thing
        // sweep examines (and pinning a sub-object inside that block
        // wouldn't keep the outer block alive — symptom: GC reclaims
        // the image after the first collect, see #47).
        const base = Number(this.exports.whisk_gc_reserve(BigInt(image_size)));
        new Uint8Array(this.memory.buffer, base, image_size)
            .set(u8.subarray(16, 16 + image_size));

        // mem.buffer can detach if the alloc grew memory; re-derive views.
        const memDV = new DataView(this.memory.buffer);

        const af = new DataView(u8.buffer, u8.byteOffset + 16 + image_size,
                                 4 * n_addr_fixups);
        for (let i = 0; i < n_addr_fixups; i++) {
            const off = af.getUint32(i * 4, true);
            const cur = memDV.getBigUint64(base + off, true);
            memDV.setBigUint64(base + off, cur + BigInt(base), true);
        }
        const lf = new DataView(u8.buffer,
            u8.byteOffset + 16 + image_size + 4 * n_addr_fixups,
            8 * n_layout_fixups);
        for (let i = 0; i < n_layout_fixups; i++) {
            const off = lf.getUint32(i * 8 + 0, true);
            const tag = lf.getUint32(i * 8 + 4, true);
            memDV.setBigUint64(base + off, BigInt(layoutAddrs[tag]), true);
        }

        const rootAddr = base + root_offset;
        this.pin(rootAddr);
        return this.handle(rootAddr);
    }

    // ── wasm-side reads back to JS (the 4 return shapes mirror Rust) ──

    /** Read `len` bytes at `ptr` as a Uint8Array (copies into a fresh buffer). */
    readBytes(ptr, len) {
        const off = Number(ptr), n = Number(len);
        return new Uint8Array(this.memory.buffer.slice(off, off + n));
    }

    /** Read `len` bytes at `ptr` as a UTF-8 string. */
    readString(ptr, len) {
        const off = Number(ptr), n = Number(len);
        return TEXT_DEC.decode(new Uint8Array(this.memory.buffer, off, n));
    }

    /** Slice wasm memory as an ArrayBuffer (copy). Useful for Marshal-decoded results. */
    readSlice(ptr, len) {
        const off = Number(ptr);
        return this.memory.buffer.slice(off, off + Number(len));
    }
}

// ── Public entry point ───────────────────────────────────────────────

/**
 * Instantiate a Whisk-compiled wasm module.
 *
 * @param {Uint8Array|ArrayBuffer|WebAssembly.Module} wasm — bytes or a precompiled Module
 * @param {object} [opts]
 * @param {object} [opts.manifest] — emitted alongside compile_to_wasm; required for uploadImage / closureImage
 * @param {Uint8Array|ArrayBuffer} [opts.closureImage] — closure capture image
 *        emitted by `compile_to_wasm` when `f` is a callable struct with fields.
 *        Auto-uploaded and installed into `__whisk_closure_root` before this
 *        promise resolves; the user-facing entry then takes only the user
 *        args (no closure-pointer plumbing required).
 * @param {object} [opts.imports] — extra import objects (e.g. `{ env: {...} }`); overrides defaults per-name
 * @returns {Promise<WhiskInstance>}
 */
async function instantiate(wasm, opts = {}) {
    const module = wasm instanceof WebAssembly.Module
        ? wasm
        : new WebAssembly.Module(wasm instanceof Uint8Array ? wasm : new Uint8Array(wasm));

    // We need a reference cell that env imports can dereference AFTER
    // instantiation completes (instance must exist before its imports
    // can read its own memory).
    const instanceRef = { value: null };
    const env = _buildEnv(module, instanceRef, (opts.imports && opts.imports.env) || {});
    const fullImports = Object.assign({}, opts.imports, { env });
    const instance = new WebAssembly.Instance(module, fullImports);
    instanceRef.value = instance;
    const wi = new WhiskInstance(instance, opts.manifest);

    // Auto-install closure image, if any. The wasm exports
    // `whisk_set_closure_root(root)`; the wrapper entry loads from it.
    if (opts.closureImage) {
        if (!instance.exports.whisk_set_closure_root) {
            throw new Error('whisk: closureImage was passed but the wasm has no `whisk_set_closure_root` export — was the function compiled with a singleton (no captures)?');
        }
        const handle = wi.uploadImage(opts.closureImage);
        instance.exports.whisk_set_closure_root(handle.addr);
        // Keep the handle pinned for the instance lifetime; the user has
        // no reason to release it (the closure must live as long as the
        // instance can be called). Stash it on the instance to keep
        // FinalizationRegistry from collecting + unpinning it.
        wi._closureHandle = handle;
    }

    return wi;
}

export { instantiate, WhiskInstance, WhiskHandle, ByteReader };
export const Whisk = { instantiate, WhiskInstance, WhiskHandle, ByteReader };
export default Whisk;
