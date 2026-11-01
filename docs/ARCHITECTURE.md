# Whisk Architecture — the end-state design

This document is the binding architecture for Whisk as a **full Julia →
WebAssembly compiler**: any type-stable Julia program compiles to wasm with
native-grade code quality, dynamic Julia (Any-containers, runtime dispatch,
exceptions) works correctly under a closed-world assumption, and failures
are compile-time diagnostics, not runtime traps. Tactical deviations from
this document are allowed only with a `# DEBT(<issue>):` marker and a plan
to remove them.

## Design principles

1. **Reuse Julia's compiler, own the runtime.** Inference, optimization and
   codegen are Julia's own (via GPUCompiler's AbstractInterpreter + LLVM).
   Whisk never re-implements language semantics; it owns everything the
   host VM normally provides at runtime: allocation, GC, type identity,
   dispatch tables, exceptions, and the data image. Runtime code is plain
   Julia compiled through the same pipeline (`COMPILER_RT_HELPERS`) — IR
   builders are reserved for the few leaf primitives that cannot be
   expressed in Julia (allocator internals, mark loop).
2. **Closed-world, whole-program AOT** (the `juliac --trim` model). The
   unit of compilation is an entry-point set. Reachability over the typed
   call graph defines the universe of MethodInstances, types, and
   constants. Everything outside it does not exist in the artifact.
3. **One type system.** The unified type descriptor (layout prefix + type
   metadata + field records, `obj[-8]` is the tag) is the single runtime
   representation of a Julia DataType. Anything that needs runtime type
   information — GC marking, `typeof`/`isa`, `getfield`, dispatch,
   serialization — reads the descriptor. No parallel registries.
4. **Constants are serialized, never scavenged.** Compile-time constants
   (literals, singletons, type instances, interned Symbols, default
   attribute dicts, …) are collected from the typed IR — where they are
   real Julia objects — and emitted as a **static data image**: a wasm data
   segment laid out by the image builder with descriptor-tagged objects and
   link-time relocations. Scraping host addresses out of LLVM constants is
   forbidden as a primary mechanism (it survives only as a verifier that
   the image caught everything).
5. **Diagnostics over traps.** If the closed world cannot resolve a
   callsite, the compile fails with the Julia source location and the
   inferred signature. Runtime traps exist only for genuine runtime errors
   (bounds, OOM, type errors on data) and always report file:line through
   `js_console` before halting.
6. **Every pass has a contract.** Each lowering pass states its
   post-condition; `validate_for_wasm!` enforces the final closed IR
   contract (no `julia.*` intrinsics, no `jl_*`/`ijl_*` externs outside the
   runtime whitelist, no variadic calls, no addrspace ≠ 0). A pass that
   cannot establish its post-condition errors out — it does not leave work
   for a later pass to discover.
7. **Intercept at the highest level that still exists.** When something
   can't compile, fix it at the topmost of these layers — never lower:
   1. **Method overlay** (`quirks.jl`, GPUCompiler method table) — when a
      Julia *method* is the blocker (env-coupled primitives: atomics on
      single-threaded wasm). The package being compiled is never edited.
   2. **Runtime symbol implemented in Julia** (`COMPILER_RT_HELPERS`) —
      when codegen emits a call to a named C runtime function with an
      implementable ABI (`ijl_box_*`, `ijl_get_nth_field_checked`,
      `jl_genericmemory_copyto`). The implementation is plain Julia,
      compiled through the pipeline and renamed to the C symbol.
   3. **Call-site rewrite** — only for codegen-emitted builtins and
      runtime entries whose semantics depend on call-site-static facts
      that no single function can capture: `jl_apply_generic` (closed-
      world dispatch is inherently a compiler transformation) and
      `jl_f_tuple` (result type depends on arity → per-arity
      specialization of the Julia helper `whisk_rt_tuple`). Even here the
      semantics live in a sub-compiled Julia function; the LLVM pass only
      rewires the call. Hand-emitted IR for runtime *semantics* is a
      design violation.
   A case like `jl_f__svec_ref` shows the layers' boundary: the function
   is trivially expressible in Julia, but its *data* (`T.parameters`
   svecs) isn't in the wasm image yet — so it traps loudly until the
   static data image (phase 4) ships type objects, at which point it
   moves up to layer 2.

## Pipeline

```
entry points (f, argtypes)…
   ↓  [1] REACHABILITY: typed call graph walk (Julia inference via
          GPUCompiler interpreter). Output: MethodInstance set +
          devirtualized callsite table + dynamic-callsite table +
          constant root set.
   ↓  [2] CODEGEN: one LLVM module for the whole MI set (no reactive
          per-callsite sub-compiles). Julia optimizer pipeline as-is.
   ↓  [3] RUNTIME BINDING: link Whisk runtime (plain-Julia helpers),
          lower allocation/GC intrinsics to whisk_gc_alloc_typed with
          unified descriptors, define typetag externs.
   ↓  [4] DATA IMAGE: serialize the constant root set via the image
          builder → data segment + relocation table; rewrite IR
          references to image offsets. (Verifier: no host-VM addresses
          survive in any constant.)
   ↓  [5] DISPATCH: emit method tables for the dynamic-callsite table
          (call_indirect through the function table); unresolvable
          sites are compile errors.
   ↓  [6] EXCEPTIONS: lower Julia's handler model onto wasm-EH
          (exnref); `jl_throw` → `throw`, handler ranges → catch.
   ↓  [7] CONTRACT CHECK: validate_for_wasm! enforces the closed IR
          contract. Violations are errors, not warnings, in CI.
   ↓  [8] EMIT + LINK: LLVM wasm backend → wasm-ld (vendored LLD).
          Exports: entry points + runtime ABI (whisk_gc_*, image API).
```

Current code vs. this pipeline:
- [2] is partially reactive today (`lower_invokes!` fixpoint sub-compiles at
  `ijl_invoke` sites). DEBT: replace with reachability-driven single-module
  codegen; sub-compiles remain only as the mechanism, driven by the
  phase-[1] worklist instead of IR scraping.
- [4] is today's `rewrite_host_pointer_constants!` probing. DEBT: invert —
  the typed-IR constant walk feeds the image builder; probing becomes the
  phase-[7] verifier ("no unexplained host addresses").
- [5] is today a diagnostic trap (`lower_dynamic_dispatch!`). DEBT: method
  tables per the design below.
- [6] is today fast-path-only (`lower_exception_intrinsics!`). DEBT:
  wasm-EH lowering.

## Dispatch design (phase 5)

For each dynamic callsite (a generic-function call inference left as
`jl_apply_generic`/widened `invoke`), reachability restricted the possible
argument types to a finite set `S` (closed world):

- `|S| = 1`: devirtualize to a direct call (inference usually already did).
- small `|S|`: inline tag-switch (compare `obj[-8]` against descriptors —
  exactly what union-split isa chains already look like) → direct calls.
- general case: per-generic-function **method table**: sorted array of
  (signature tag tuple → function-table index); a plain-Julia runtime
  helper performs the lookup; calls go through `call_indirect`
  (`--export-table` already wired for closures). Descriptors get a
  `dispatch_id` slot for O(1) keying.
- A callsite whose `S` is unbounded (e.g. `Any` from external input that
  never narrows) is a compile error listing the callsite; the escape hatch
  is an explicit user-provided type list (mirrors `Base.Experimental
  @overlay`-style opt-in, never silent).

## Exceptions design (phase 6)

Julia lowers `try/catch` to `except_enter` (setjmp-like) + handler
push/pop; native unwinding is longjmp. On wasm the equivalent control
transfer is wasm-EH: `jl_throw(exc)` becomes `throw $whisk_exc` with the
exception object as payload; each `except_enter` region becomes a
`try`/`catch $whisk_exc` block; `ijl_current_exception` reads the caught
payload from a per-task slot. LLVM expresses this via `invoke` +
`llvm.wasm.throw` / catchpad funclets, so the lowering pass restructures
the CFG from Julia's handler-range model into funclet form. Engines: V8,
SpiderMonkey, JSC all ship wasm-EH (exnref) — no fallback path needed.
`finally` falls out of catch-all + rethrow.

## Performance doctrine

- Type-stable code must compile to the same -O2 IR Julia produces natively
  (we run the same optimizer; the ladder asserts numeric parity, perf
  rungs assert no boxing in hot loops).
- Boxing/dispatch/EH are correct-but-cold paths; their cost is measured
  (ladder perf rungs), never on the type-stable path.
- GC: precise marking via descriptors (done). Next levers, profile-driven
  only: bump-pointer nursery with sticky mark bits; JS-side idle-time
  collection scheduling. No GC work without a benchmark.
- Data image loads are zero-copy: the segment is laid out in final form;
  relocations are applied by wasm-ld, not at startup.

## Naming & style

Follows the repository owner's conventions: no `_`-prefixed function
names (visibility = exported or not); `const` globals UPPERCASE; multiple
dispatch over name-mangled variants; GeometryBasics types in geometry
code. Existing `_`-prefixed internals are DEBT to be renamed as files are
touched.

## Testing doctrine

- `Pkg.test("Whisk")` — per-pass unit tests + end-to-end regression
  catalogue. Must stay green on every commit.
- `ladder/run.jl` (WhiskDev) — the Makie-ladder: integration rungs ordered
  by how deep into Makie-shaped territory they reach. New capability work
  starts by adding a red rung.
- Every fixed miscompile gets a rung or a unit test reproducing it.
