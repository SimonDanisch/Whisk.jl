// Whisk JS-side handle helper.
//
// A "handle" is a wasm linear-memory address (UInt64) returned by a
// Whisk function that called `Whisk.pin_value`. The address points
// at a heap-allocated `Ref{T}` whose lifetime is anchored by a
// matching `whisk_pin` registration. Until JS calls `whisk_unpin`,
// the wasm GC can't reclaim it.
//
// Manually balancing pin/unpin is error-prone. This helper wraps the
// address in a JS object whose lifetime drives the unpin via
// FinalizationRegistry: when V8 garbage-collects the JS wrapper,
// the registry callback fires `whisk_unpin(addr)` and the wasm GC
// can collect on its next sweep.
//
// Usage:
//
//   import { WhiskHandles } from './whisk_handle.mjs';
//   const handles = new WhiskHandles(instance.exports);
//
//   // Decode-and-pin happens inside the wasm function (e.g.,
//   // SimiMaps's `init_graph` calls `Whisk.pin_value` internally).
//   // Wrap the returned address:
//   const graph = handles.wrap(exp.init_graph(blobPtr, blobLen));
//
//   // Pass to subsequent calls:
//   exp.route_query(..., graph.addr, params.addr);
//
//   // When `graph` is JS-GC'd (or you call .release()), the
//   // wasm-side address is unpinned and becomes collectable.

export class WhiskHandles {
    constructor(exports) {
        this._exports = exports;
        // FinalizationRegistry's held-value is what we get in the
        // callback. We pass the raw address; the registry holds it
        // weakly with respect to the wrapper.
        this._registry = new FinalizationRegistry((addr) => {
            try {
                this._exports.whisk_unpin(BigInt(addr));
            } catch (e) {
                // If the wasm instance has been torn down, ignore.
                console.warn("WhiskHandles: unpin failed", e);
            }
        });
    }

    /**
     * Wrap a wasm-returned address in a handle object.
     *
     * @param {bigint|number} addr Address returned by a Whisk fn that pinned it.
     * @returns {{ addr: bigint, release(): void }} Handle wrapper.
     */
    wrap(addr) {
        const addrBig = typeof addr === 'bigint' ? addr : BigInt(addr);
        // Plain object so it has identity (FinalizationRegistry needs
        // a real reference to track). Frozen so callers don't add
        // bookkeeping that would defeat the registry.
        const handle = Object.freeze({
            addr: addrBig,
            release: () => {
                if (this._released.has(addrBig)) return;
                this._released.add(addrBig);
                this._registry.unregister(handle);
                this._exports.whisk_unpin(addrBig);
            },
        });
        this._registry.register(handle, Number(addrBig), handle);
        return handle;
    }

    // Per-instance set of addresses already manually-released so
    // double-release is a no-op.
    get _released() {
        if (!this.__released) this.__released = new Set();
        return this.__released;
    }
}

// Convenience factory — most callers want exactly the above shape.
export function makeWhiskHandles(exports) {
    return new WhiskHandles(exports);
}
