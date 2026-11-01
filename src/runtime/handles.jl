## Generic typed-handle helpers for keeping wasm-side values alive
## across JS/wasm calls.
##
## Pattern: a wasm function decodes/computes some long-lived state,
## boxes it in a `Ref{T}` (whose mutable wrapper has a stable heap
## address), pins the address with `whisk_pin` so the wasm GC can't
## reclaim it, and returns the address as a `UInt64`. JS holds the
## opaque address — typically wrapping it with `FinalizationRegistry`
## so that JS-GC of the wrapper triggers a `whisk_unpin` (see
## `jsinterop/whisk_handle.mjs`). Subsequent wasm calls take the
## address as an argument and recover the typed value via
## `unwrap_value`.
##
## Why `Ref{T}` and not a custom mutable handle struct: `Ref{T}` is
## exactly the right thing — Julia's standard single-field mutable
## wrapper around a value. Works for any T (immutable structs with
## reference fields, mutable types, etc.) because `Base.RefValue{T}`
## is itself mutable, so `pointer_from_objref` accepts it.
##
## Recovery uses `unsafe_load(Ptr{Ref{T}})` rather than
## `unsafe_pointer_to_objref(p)::T` or `ccall(:jl_value_ptr, T, …)` —
## the latter both trip Whisk's no-type-tags constraint (typeassert
## traps; mutable-T ccall infers Bottom). `unsafe_load` is a plain
## `Core.Intrinsics.pointerref` that copies the wrapper's bytes; the
## inner immutable T's reference fields (Vectors, Strings, …) stay
## valid as long as the outer Ref's address is JS-pinned.

"""
    Whisk.pin_value(x) → UInt64

Box `x` in a mutable `Ref{typeof(x)}`, pin the wrapper's heap
address with the wasm GC, and return that address. The caller (JS)
must eventually `whisk_unpin(addr)` — typically by attaching a
`FinalizationRegistry` to a JS wrapper object so V8 GC drives the
unpin.

Recover with `Whisk.unwrap_value(T, addr)` on subsequent calls.

This is the canonical pattern for cross-call cached state in Whisk:
no decode on every call, no internal "current session" state in
wasm — JS owns the lifetime, wasm just exposes a typed address.
"""
@inline function pin_value(x)
    ref = Ref(x)
    addr = UInt64(UInt(pointer_from_objref(ref)))
    ccall(:whisk_pin, Cvoid, (UInt64,), addr)
    return addr
end

"""
    Whisk.unwrap_value(::Type{T}, addr::UInt64) → T

Recover a value previously pinned via `pin_value(x::T)`. Returns a
value-copy of T. For immutable T with reference fields (Vector,
String, etc.), the inner pointers remain alive as long as `addr` is
still JS-pinned.

Reads via `unsafe_load(Ptr{Ref{T}})` to avoid the typeassert path
that would require runtime type tags Whisk doesn't have.
"""
@inline function unwrap_value(::Type{T}, addr::UInt64) where T
    ref = unsafe_load(Ptr{Base.RefValue{T}}(addr))
    return ref[]
end
