// deno-fmt-ignore-file
// deno-lint-ignore-file
// This code was bundled using `deno bundle` and it's not recommended to edit it manually

const TEXT_ENC = new TextEncoder();
const TEXT_DEC = new TextDecoder('utf-8');
class ByteReader {
    constructor(buf){
        if (buf instanceof Uint8Array) {
            this.view = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
        } else if (buf instanceof DataView) {
            this.view = buf;
        } else {
            this.view = new DataView(buf);
        }
        this.pos = 0;
    }
    u8() {
        const v = this.view.getUint8(this.pos);
        this.pos += 1;
        return v;
    }
    i8() {
        const v = this.view.getInt8(this.pos);
        this.pos += 1;
        return v;
    }
    u16() {
        const v = this.view.getUint16(this.pos, true);
        this.pos += 2;
        return v;
    }
    i16() {
        const v = this.view.getInt16(this.pos, true);
        this.pos += 2;
        return v;
    }
    u32() {
        const v = this.view.getUint32(this.pos, true);
        this.pos += 4;
        return v;
    }
    i32() {
        const v = this.view.getInt32(this.pos, true);
        this.pos += 4;
        return v;
    }
    u64() {
        const v = this.view.getBigUint64(this.pos, true);
        this.pos += 8;
        return Number(v);
    }
    i64() {
        const v = this.view.getBigInt64(this.pos, true);
        this.pos += 8;
        return Number(v);
    }
    u64big() {
        const v = this.view.getBigUint64(this.pos, true);
        this.pos += 8;
        return v;
    }
    i64big() {
        const v = this.view.getBigInt64(this.pos, true);
        this.pos += 8;
        return v;
    }
    f32() {
        const v = this.view.getFloat32(this.pos, true);
        this.pos += 4;
        return v;
    }
    f64() {
        const v = this.view.getFloat64(this.pos, true);
        this.pos += 8;
        return v;
    }
    bool() {
        return this.u8() !== 0;
    }
    vector(elem) {
        const n = this.u32();
        const out = new Array(n);
        for(let i = 0; i < n; i++)out[i] = elem(this);
        return out;
    }
    string() {
        const n = this.u32();
        const b = new Uint8Array(this.view.buffer, this.view.byteOffset + this.pos, n);
        this.pos += n;
        return TEXT_DEC.decode(b);
    }
}
class WhiskHandle {
    constructor(addr, registry, exportsRef){
        this.addr = typeof addr === 'bigint' ? addr : BigInt(addr);
        this._reg = registry;
        this._exp = exportsRef;
        this._released = false;
        Object.freeze(this);
    }
    release() {
        if (this._released) return;
        this._reg.unregister(this);
        try {
            this._exp.value.whisk_unpin(this.addr);
        } catch (e) {
            console.warn('WhiskHandle.release: unpin failed', e);
        }
    }
}
function _makeHandleFactory(exportsRef) {
    const reg = new FinalizationRegistry((addr)=>{
        try {
            exportsRef.value.whisk_unpin(BigInt(addr));
        } catch (e) {}
    });
    return (addr)=>{
        const h = new WhiskHandle(addr, reg, exportsRef);
        reg.register(h, Number(h.addr), h);
        return h;
    };
}
function _buildEnv(module, instanceRef, callerEnv = {}) {
    const env = {};
    const memOf = ()=>instanceRef.value && instanceRef.value.exports.memory;
    const N = (b)=>typeof b === 'bigint' ? Number(b) : b;
    const defaults = {
        memset: (p, v, n)=>{
            const m = memOf();
            if (!m) return p;
            new Uint8Array(m.buffer).fill(Number(v) & 0xff, N(p), N(p) + N(n));
            return p;
        },
        memcpy: (d, s, n)=>{
            const m = memOf();
            if (!m) return d;
            const dn = N(d), sn = N(s), ln = N(n);
            new Uint8Array(m.buffer, dn, ln).set(new Uint8Array(m.buffer, sn, ln));
            return d;
        },
        memmove: (d, s, n)=>{
            const m = memOf();
            if (!m) return d;
            new Uint8Array(m.buffer).copyWithin(N(d), N(s), N(s) + N(n));
            return d;
        },
        js_console: (level, ptr, len)=>{
            const m = memOf();
            if (!m) return;
            const txt = TEXT_DEC.decode(new Uint8Array(m.buffer, N(ptr), N(len)));
            (level === 2 ? console.error : level === 1 ? console.warn : console.log).call(console, '[wasm] ' + txt);
        }
    };
    for (const desc of WebAssembly.Module.imports(module)){
        if (desc.module !== 'env' || desc.kind !== 'function') continue;
        env[desc.name] = callerEnv[desc.name] !== undefined ? callerEnv[desc.name] : defaults[desc.name] !== undefined ? defaults[desc.name] : (...args)=>{
            throw new Error(`whisk: unstubbed wasm import env.${desc.name}(${args.length} args)`);
        };
    }
    return env;
}
class WhiskInstance {
    constructor(instance, manifest){
        this._instance = instance;
        this.exports = instance.exports;
        this.memory = instance.exports.memory;
        this.manifest = manifest || null;
        this._exportsRef = {
            value: instance.exports
        };
        this._handle = _makeHandleFactory(this._exportsRef);
        this._layoutAddrs = null;
    }
    malloc(n) {
        return this.exports.whisk_gc_alloc(BigInt(n));
    }
    pin(addr) {
        this.exports.whisk_pin(typeof addr === 'bigint' ? addr : BigInt(addr));
    }
    unpin(addr) {
        this.exports.whisk_unpin(typeof addr === 'bigint' ? addr : BigInt(addr));
    }
    collect() {
        this.exports.whisk_gc_collect();
    }
    handle(addr) {
        return this._handle(addr);
    }
    get heapU8() {
        return new Uint8Array(this.memory.buffer);
    }
    get heapDV() {
        return new DataView(this.memory.buffer);
    }
    layoutAddrOf(tag) {
        if (!this.exports.__whisk_layout_addr) {
            throw new Error('whisk: wasm has no __whisk_layout_addr export — was layout_tags passed to compile_to_wasm?');
        }
        return Number(this.exports.__whisk_layout_addr(BigInt(tag)));
    }
    _resolveLayouts() {
        if (this._layoutAddrs) return this._layoutAddrs;
        if (!this.manifest || !this.manifest.tags) {
            throw new Error('whisk: instance has no manifest; pass {manifest} to instantiate()');
        }
        this._layoutAddrs = this.manifest.tags.map((t)=>this.layoutAddrOf(t.tag));
        return this._layoutAddrs;
    }
    uploadString(s) {
        const bytes = TEXT_ENC.encode(s);
        return this.uploadBytes(bytes);
    }
    uploadBytes(u8) {
        if (!(u8 instanceof Uint8Array)) u8 = new Uint8Array(u8);
        const ptr = this.malloc(u8.byteLength);
        new Uint8Array(this.memory.buffer, Number(ptr), u8.byteLength).set(u8);
        return {
            ptr,
            len: u8.byteLength
        };
    }
    uploadF32Array(arr) {
        const ta = arr instanceof Float32Array ? arr : new Float32Array(arr);
        const u8 = new Uint8Array(ta.buffer, ta.byteOffset, ta.byteLength);
        const { ptr  } = this.uploadBytes(u8);
        return {
            ptr,
            len: ta.length
        };
    }
    uploadF64Array(arr) {
        const ta = arr instanceof Float64Array ? arr : new Float64Array(arr);
        const u8 = new Uint8Array(ta.buffer, ta.byteOffset, ta.byteLength);
        const { ptr  } = this.uploadBytes(u8);
        return {
            ptr,
            len: ta.length
        };
    }
    uploadI32Array(arr) {
        const ta = arr instanceof Int32Array ? arr : new Int32Array(arr);
        const u8 = new Uint8Array(ta.buffer, ta.byteOffset, ta.byteLength);
        const { ptr  } = this.uploadBytes(u8);
        return {
            ptr,
            len: ta.length
        };
    }
    uploadI64Array(arr) {
        const ta = arr instanceof BigInt64Array ? arr : BigInt64Array.from(arr, BigInt);
        const u8 = new Uint8Array(ta.buffer, ta.byteOffset, ta.byteLength);
        const { ptr  } = this.uploadBytes(u8);
        return {
            ptr,
            len: ta.length
        };
    }
    uploadImage(blob) {
        const u8 = blob instanceof Uint8Array ? blob : new Uint8Array(blob);
        const hdr = new DataView(u8.buffer, u8.byteOffset, 16);
        const image_size = hdr.getUint32(0, true);
        const root_offset = hdr.getUint32(4, true);
        const n_addr_fixups = hdr.getUint32(8, true);
        const n_layout_fixups = hdr.getUint32(12, true);
        const layoutAddrs = this._resolveLayouts();
        const base = Number(this.malloc(image_size));
        new Uint8Array(this.memory.buffer, base, image_size).set(u8.subarray(16, 16 + image_size));
        const memDV = new DataView(this.memory.buffer);
        const af = new DataView(u8.buffer, u8.byteOffset + 16 + image_size, 4 * n_addr_fixups);
        for(let i = 0; i < n_addr_fixups; i++){
            const off = af.getUint32(i * 4, true);
            const cur = memDV.getBigUint64(base + off, true);
            memDV.setBigUint64(base + off, cur + BigInt(base), true);
        }
        const lf = new DataView(u8.buffer, u8.byteOffset + 16 + image_size + 4 * n_addr_fixups, 8 * n_layout_fixups);
        for(let i = 0; i < n_layout_fixups; i++){
            const off = lf.getUint32(i * 8 + 0, true);
            const tag = lf.getUint32(i * 8 + 4, true);
            memDV.setBigUint64(base + off, BigInt(layoutAddrs[tag]), true);
        }
        const rootAddr = base + root_offset;
        this.pin(rootAddr);
        return this.handle(rootAddr);
    }
    readBytes(ptr, len) {
        const off = Number(ptr), n = Number(len);
        return new Uint8Array(this.memory.buffer.slice(off, off + n));
    }
    readString(ptr, len) {
        const off = Number(ptr), n = Number(len);
        return TEXT_DEC.decode(new Uint8Array(this.memory.buffer, off, n));
    }
    readSlice(ptr, len) {
        const off = Number(ptr);
        return this.memory.buffer.slice(off, off + Number(len));
    }
}
async function instantiate(wasm, opts = {}) {
    const module = wasm instanceof WebAssembly.Module ? wasm : new WebAssembly.Module(wasm instanceof Uint8Array ? wasm : new Uint8Array(wasm));
    const instanceRef = {
        value: null
    };
    const env = _buildEnv(module, instanceRef, opts.imports && opts.imports.env || {});
    const fullImports = Object.assign({}, opts.imports, {
        env
    });
    const instance = new WebAssembly.Instance(module, fullImports);
    instanceRef.value = instance;
    return new WhiskInstance(instance, opts.manifest);
}
export { instantiate as instantiate, WhiskInstance as WhiskInstance, WhiskHandle as WhiskHandle, ByteReader as ByteReader };
const Whisk = {
    instantiate,
    WhiskInstance,
    WhiskHandle,
    ByteReader
};
export { Whisk as Whisk };
export { Whisk as default };

