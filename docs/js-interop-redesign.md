# Whisk JS↔Wasm Interop Redesign

## Goal

Replace Whisk's current "JS places data at fixed linear-memory offsets +
ad-hoc heap reset" interop with the production-grade model used by every
real wasm-targeting GC'd language (Go, Kotlin/Wasm) and adapted from
Emscripten's malloc-export model:

> **The wasm module owns linear memory entirely. JS uses the wasm
> allocator for every byte it writes into linear memory. Long-lived
> JS-held pointers are explicitly pinned as GC roots. Mark+sweep is the
> only mechanism that keeps the heap bounded.**

When this is done, `whisk_reset_heap` does not exist and SimiMaps does
not know about heap layout.

## Why this needs a spec

Past attempts have repeatedly slid back to handrolled workarounds:
JS-driven per-call reset, JS-driven init-time reset, Julia-side
per-call reset inside `route_full`. Each "fix" was a variant of the
same anti-pattern. This spec exists so the work can be checked against
a fixed external reference, not against the implementation's own
narrative.

## Anti-patterns (forbidden — failing any of these fails the phase)

1. **No fixed-offset placement in JS.** No `const G_OFF = 16 * 1024 * 1024`
   or any equivalent. Every linear-memory address JS writes to comes
   from `whisk_gc_alloc` (or, for the output buffer, also from
   `whisk_gc_alloc`).
2. **No `whisk_reset_heap`.** Not as an export, not as a runtime
   symbol, not as an LLVM function. `_emit_reset_heap!` is deleted.
   The `--export=whisk_reset_heap` flag is removed from
   `compile_to_wasm`.
3. **No per-call reset of `_whisk_heap_next` from anywhere** — not
   from JS, not from Julia entry points, not from Whisk-emitted IR
   prologues.
4. **No JS-side knowledge of `__heap_base`** beyond what's strictly
   needed for the allocator's own implementation. The wrapper does not
   read it to "know where to put data."
5. **No "for now" / "temporary" hacks** that leave any of the above
   in place "until later."
6. **No host-side trickery to compensate for missing GC roots.**
   If a JS-held pointer can be reclaimed prematurely, the answer is to
   pin it with `whisk_pin`, not to hold it briefly or hope for the best.

## Architecture

### Components

1. **JS-roots table inside the wasm runtime.**
   - Fixed-size static global in `runtime/gc.jl`:
     `@_whisk_js_roots` — `[256 x i64] zeroinitializer`,
     `@_whisk_js_roots_len` — current count.
   - Fixed-size matches JNI's global-reference table and Lua's
     pseudo-stack. Apps pin a handful of long-lived buffers (graph,
     params); 256 slots is generous. Overflow traps with a diagnostic
     so misuse is caught loudly rather than silently corrupting the
     heap. If a real app ever wants more, this is mechanical to
     replace with growth via `whisk_gc_alloc` later.

2. **`whisk_pin(ptr: i64)` and `whisk_unpin(ptr: i64)` exports.**
   - Emitted in `runtime/gc.jl`.
   - `whisk_pin` appends the pointer to the JS-roots list, growing the
     backing array if needed.
   - `whisk_unpin` finds the pointer and removes it (swap-with-last, no
     order preserved).
   - Both are wasm-exported (`wasm-export-name`) and pinned in
     `@llvm.used`.

3. **`_whisk_gc_mark` walks the JS-roots list.**
   - In addition to walking `_whisk_gc_frames`, the mark phase also
     walks `_whisk_js_roots_buf[0 .. _whisk_js_roots_len)` and calls
     `_whisk_mark_obj_at_addr` on each.

4. **JS wrapper consumes the allocator.**
   - SimiMaps' `wasm_router.jl` (the JS bridge) and the test harness:
     - On wasm instantiate, allocate graph + params blobs via
       `instance.exports.whisk_gc_alloc(blob_len)`.
     - Write blob bytes into those wasm-allocated regions.
     - Call `whisk_pin(graph_ptr)` and `whisk_pin(params_ptr)` to root
       them across calls.
     - Allocate the output buffer per call via `whisk_gc_alloc(OUT_CAP)`
       — no pinning needed; the buffer is reachable from the Julia
       stack frame for the duration of the call.
     - Pass the pointers as before to `route_full`/`astar_full`.
     - On JS-side cleanup (page unload, app exit), call `whisk_unpin`
       on the pinned pointers. Optional in tests; required if real apps
       want to reload/reinit.

### What the GC's mark+sweep guarantees after this change

- **Cross-call stability**: the pinned graph + params survive every
  collect, because `_whisk_gc_mark` finds them via the JS-roots list.
- **Steady-state heap**: each `route_full` call allocates ~K bytes of
  internal Julia state (decoded RoutingGraph copies, A* heaps,
  encoder buffer). All of those are in `_whisk_gc_frames` during the
  call and dead afterwards. The next call's
  `whisk_ensure_capacity → whisk_gc_collect` sweeps them into the
  free list. Subsequent allocs are served from the free list before
  bumping. Memory growth stops at the high-water mark of a single
  call's working set.
- **No heap-vs-JS-region collision**: the only data in linear memory
  is what `whisk_gc_alloc` placed there. JS never picks an address.

## Phases

Each phase has a deliverable, an external check, and an explicit pass/fail
gate. **Do not advance to the next phase until the current one passes
its `superpowers:code-reviewer` review.**

### Phase 1 — Pin/unpin runtime support

**Deliverable**: `runtime/gc.jl` emits `whisk_pin` / `whisk_unpin` wasm
exports + extends `_whisk_gc_mark` to walk the JS-roots list.

**External check**: dispatch `superpowers:code-reviewer` with the
following brief:

> "Review `runtime/gc.jl` against `docs/js-interop-redesign.md`
> Phase 1. Verify: (a) `whisk_pin` and `whisk_unpin` are emitted with
> `wasm-export-name` attributes and `@llvm.used` pinning; (b) the
> JS-roots table is the documented fixed-size 256-slot static global
> with overflow trapping; (c) `_whisk_gc_mark` reads the current
> `_whisk_js_roots_len` and calls `_whisk_mark_obj_at_addr` for each
> live entry, AFTER walking pgcstack; (d) sub-compile dumps of
> `_whisk_gc_mark` and `whisk_pin`/`whisk_unpin` show the expected
> control flow without `gpu_report_exception` stubs; (e) no
> regressions in existing Whisk tests. Report concrete file:line
> findings, not summaries."

**Pass gate**: reviewer reports zero spec violations.

### Phase 2 — Delete `whisk_reset_heap`

**Deliverable**:
- Delete `_emit_reset_heap!` from `runtime/gc.jl`.
- Remove the call to `_emit_reset_heap!` from `emit_wasm_runtime!`.
- Remove `--export=whisk_reset_heap` from `compile_to_wasm` /
  wherever the linker flags are assembled.
- Search the whole tree for any remaining mention of `whisk_reset_heap`.
  Zero hits in `src/`. References in `docs/` are allowed only if they
  describe the historical pattern as obsolete.

**External check**: dispatch `superpowers:code-reviewer`:

> "Verify `whisk_reset_heap` is fully removed from
> `dev/Whisk/src/` and `dev/SimiMaps/src/`. Report any surviving
> reference. Also verify no JS code in either repo calls
> `instance.exports.whisk_reset_heap`."

**Pass gate**: reviewer confirms zero references in source.

### Phase 2.5 — Real GC, host-driven safepoint

**Problem found during Phase 3 testing**: with `whisk_ensure_capacity`
calling `whisk_gc_collect` mid-allocation, mark phase ran with the
pgcstack empty (Julia 1.11+'s frontend doesn't emit `julia.new_gc_frame`
for our wasm target without `uses_julia_runtime = true`, and that
flag opens up cascading Julia-vs-GPUCompiler IR incompatibilities
that segfault LLVM's linker).

**Fix**: separate concerns — `whisk_ensure_capacity` only `memory.grow`s,
and the host triggers `whisk_gc_collect()` between wasm calls. At
that safepoint, pgcstack is genuinely empty (we're between calls)
so mark walks only the JS-pinned roots (the input/output buffers JS
allocated and pinned), and sweep reclaims every internal allocation
the prior call made. Heap stays bounded across many calls.

This is the same pattern Go's wasm runtime uses: GC runs at
explicit safepoints driven from JS, not preemptively in the middle
of a call. The wasm SSA register set isn't visible to mark, so a
mid-call collect would always be wrong; deferring to a known
safepoint is the right shape.

**Deliverable**:
- `whisk_ensure_capacity` in `runtime/intrinsics.jl` skips
  `whisk_gc_collect`, only calls `memory.grow`.
- `routeFull` in `wasm_router.jl` calls
  `instance.exports.whisk_gc_collect()` after reading the result
  (in a `finally` so failures still GC).
- The 1000-call test harness mirrors the same pattern.

**External check**: run `Pkg.test('SimiMaps')`. The
`1000 sequential route_full calls — stable + bounded heap` testset
passes:
- Same hash on every checkpoint (no data corruption).
- Same `written` count on every checkpoint.
- Memory stabilises after the first ~10 calls (the allocator's
  free-list develops the chunk-size profile this workload needs).
- No memory leak — calls 100/500/1000 all see the same memory size.

**Pass gate**: SimiMaps stress test green + Whisk's 82/82 unaffected.

**Status**: SHIPPED. SimiMaps: 8/8, Whisk: 82/82. The 1000-call
stress run shows `written=98, hash=0x140baa96` consistent across
all checkpoints, peak/stable memory 80 MB.

### Phase 3 — JS wrapper switches to the allocator

**Deliverable**: in `dev/SimiMaps/src/server/wasm_router.jl`:
- Remove `G_OFF`, `P_OFF`, `O_OFF`, `OUT_CAP` fixed-offset constants.
- After `WebAssembly.instantiate`, allocate graph + params blobs via
  `instance.exports.whisk_gc_alloc(BigInt(blob.length))`. Write blob
  bytes via `new Uint8Array(mem.buffer, ptr, len)`.
- Call `instance.exports.whisk_pin(graph_ptr)` and `whisk_pin(params_ptr)`.
- For each `routeFull` call, allocate the output buffer via
  `whisk_gc_alloc`, **pin it via `whisk_pin`**, pass it to
  `route_full`, decode the bytes, then `whisk_unpin`. The output
  buffer must be a GC root because primitive `Ptr{UInt8}` arguments
  to `route_full` are not tracked by the pgcstack walker — a collect
  triggered by a Julia allocation inside the call (e.g. encoder
  buffer growth past `memory.size`) would otherwise mark the
  unpinned buffer dead and let a subsequent allocation reuse its
  region mid-call.
- Mirror the same pattern in `dev/SimiMaps/test/runtests.jl`.

**External check**: dispatch `superpowers:code-reviewer`:

> "Review `dev/SimiMaps/src/server/wasm_router.jl` and
> `dev/SimiMaps/test/runtests.jl` against `docs/js-interop-redesign.md`
> Phase 3. Confirm: (a) no fixed-offset constants remain (grep for
> `G_OFF`, `P_OFF`, `O_OFF`, `OUT_CAP`, `16 * 1024 * 1024`); (b) every
> JS write into linear memory targets a pointer obtained from
> `whisk_gc_alloc`; (c) graph + params are pinned via `whisk_pin`
> exactly once after instantiate; (d) `whisk_unpin` is called on
> teardown if the JS-side has a teardown path; (e) the output buffer
> is allocated per-call AND pinned around the call (pin → route →
> decode → unpin). Report file:line findings."

**Pass gate**: reviewer confirms zero violations.

### Phase 4 — End-to-end verification

**Deliverable**: the existing 1000-call stress test passes:
- Same hash on every checkpoint call (no data corruption).
- Memory growth stops after the first few calls and stays flat
  through call 1000.
- Peak memory under the 64 MB sanity bound.
- No `instance.exports.whisk_reset_heap` call exists in the harness
  because that export does not exist.

**External check**: dispatch `superpowers:code-reviewer`:

> "Run `Pkg.test('SimiMaps')` and confirm the
> `1000 sequential route_full calls — stable + bounded heap`
> testset passes. Inspect the test source to verify it doesn't call
> `whisk_reset_heap`. Inspect the wasm output via `wasm-objdump -x`
> (or equivalent) to confirm `whisk_reset_heap` is not in the export
> table. Report concrete numbers (peak memory, written bytes, hash
> stability) plus the confirmation."

**Pass gate**: tests pass + reviewer confirms no `whisk_reset_heap`
in the output.

## Self-check protocol per phase

Before marking a phase done:

1. Re-read the **Anti-patterns** list above. For each, name the line
   in the phase's deliverable that proves it isn't violated.
2. Dispatch `superpowers:code-reviewer` per the phase brief.
3. If the reviewer reports any violation, **revert the violation**
   (don't argue with the reviewer, don't reinterpret the spec). Re-run
   the review.
4. Only after a clean review, advance to the next phase.

## Out of scope

- Multi-memory wasm. Possibly cleaner long-term, but unsupported by
  current LLD + Julia toolchain.
- Compaction. Mark+sweep + free-list is sufficient for steady-state.
- A separate "JS heap" region. Not needed once JS uses the allocator.
- Non-Whisk targets (the wasm32 path, if it still exists).
