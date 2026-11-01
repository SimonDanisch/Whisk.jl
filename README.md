# Whisk.jl

Compile Julia → WebAssembly via the standard LLVM IR pipeline.

```
Julia source
   ↓  (GPUCompiler.AbstractInterpreter + Whisk method-table overrides)
typed CodeInfo / IRCode
   ↓  (Julia's emit_function — `GPUCompiler.compile(:llvm, job)`)
LLVM IR  (wasm32-unknown-unknown triple)
   ↓  (LLVM_full_jll's `llc` — battle-tested wasm backend)
wasm object file
   ↓  (`wasm-ld` from LLD_jll)
.wasm module + JS marshal helpers
```

The thesis: **operate on a versioned spec (LLVM IR), not on Julia's internal
typed CodeInfo**. Mature toolchain, no dialect drift between Julia versions, no
hand-rolled relooper, no Binaryen API holes.

## Status

M1, M2, M3 complete. **39 / 40 tests** green — only String runtime
indexing remains broken until M4.

## Roadmap

| | Milestone | What works | Notes |
|---|---|---|---|
| ✓ M1 | `f(x::Int32)::Int32 = x + 1` runs in V8 | end-to-end pipeline | done |
| ✓ M2 | `Vector{T}(undef, n)`, indexing | linear-memory allocation | done |
| ✓ M3 | mark-sweep GC, retains across calls | precise roots, conservative scan | done |
| M4 | `Dict`, `Set`, `BitVector` | normal stdlib, no overrides | should mostly work given M3 |
| M5 | `@jsexport`, JS marshal | end-to-end JS↔wasm round trip | typed-array based |
| M6 | A* with `BinaryMinHeap` | parity vs WAC | regression check |
| M7 | ccall validator | friendly errors for unsupported ccalls | only invalid Julia surface |

## GC architecture

```
Linear-memory layout after wasm-ld:

    0  ┌────────────────────────────────┐
       │ rodata, data segments          │
       ├────────────────────────────────┤
       │ wasm stack (grows down)        │
 __heap_base ───────────────────────────┤
       │ allocated heap (header+payload)│
       │   ↓ bump allocator             │
 _whisk_heap_next ──────────────────────┤
       │ free space                     │
    ↓  └────────────────────────────────┘
```

Each allocation has an 8-byte header: 4-byte size + 4-byte flags
(mark / free bits). The free list reuses freed chunks' payloads as
"next free" pointers.

Roots come from Julia's GC frame intrinsics (`julia.{new, push, pop}_
gc_frame`, `julia.get_gc_frame_slot`, `julia.get_pgcstack`). We
*lower* them — instead of nullifying, as a minimal implementation
might — into real linked-list operations on `_whisk_pgcstack`. That
gives us PRECISE roots (the Julia compiler tells us which locals are
GC pointers); payload bodies are scanned conservatively (every
aligned 4-byte word treated as a potential heap pointer).

`whisk_gc_collect` is exported from every wasm module so callers can
drive GC explicitly.

## ccall policy

ccall is the only invalid Julia code. Three buckets, decided at IR-validation:

- **Allowed** — `memset`, `memcpy`, `memmove`, libm math (`sqrtf`, `sinf`, …) —
  LLVM's wasm backend lowers / imports these naturally.
- **Translated** — registry-mapped to wasm equivalents (`unsafe_load`, etc.).
- **Friendly error** — anything else throws `Whisk.UnsupportedCcallError`
  pointing at the offending source line.
