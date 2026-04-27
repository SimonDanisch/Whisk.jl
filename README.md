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

Pre-M1. Scaffolding only.

## Roadmap

| | Milestone | What works | Notes |
|---|---|---|---|
| M1 | `f(x::Int32)::Int32 = x + 1` runs in V8 | end-to-end pipeline | no allocator yet |
| M2 | `Vector{Float32}(undef, n)` | bump allocator | linear-memory runtime |
| M3 | mark-sweep GC | allocations across calls retained | full GC integration |
| M4 | `Dict`, `Set`, `BitVector` | normal stdlib, no overrides | mostly free if M3 holds |
| M5 | `@jsexport`, JS marshal | end-to-end JS↔wasm round trip | typed-array based |
| M6 | A* with `BinaryMinHeap` | parity vs WAC | regression check |
| M7 | ccall validator | friendly errors for unsupported ccalls | only invalid Julia surface |

## ccall policy

ccall is the only invalid Julia code. Three buckets, decided at IR-validation:

- **Allowed** — `memset`, `memcpy`, `memmove`, libm math (`sqrtf`, `sinf`, …) —
  LLVM's wasm backend lowers / imports these naturally.
- **Translated** — registry-mapped to wasm equivalents (`unsafe_load`, etc.).
- **Friendly error** — anything else throws `Whisk.UnsupportedCcallError`
  pointing at the offending source line.
