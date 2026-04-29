# Whisk · Next Phase Plan

Self-contained plan for a fresh session. Read this first; you don't need
context from any prior conversation to act on it.

## Where Whisk is right now

**Working** (regression-tested via `Pkg.test("Whisk")`, currently 72/72):
- M1: scalar arithmetic.
- M2: `Vector{T}(undef, n)` with `setindex!` / `getindex`.
- M3: mark-sweep GC (real, not bump-only).
- M4: `Dict{UInt8, Float32}`, `Set{UInt8}`, `BitVector`, `String` codeunit access.
- M5: `sin` / `cos` / `sqrt` / `asin` / haversine end-to-end.
  - i128 helpers (`__multi3`, `__lshrti3`, `__ashlti3`) are linked in via
    `link_compiler_rt!` in `src/compiler/compilation.jl` — they're plain
    Julia functions in `src/runtime/intrinsics.jl`, sub-compiled and
    renamed to the C ABI symbol.

**Compiles but doesn't validate / link / run yet:** anything that goes
through `Vector` `push!` / `pop!` (because of `_growend!` / `unsafe_copyto!`),
which is the gateway to `BinaryMinHeap` and therefore to the real
`SimiMaps.astar`.

## The blocker

The chain `push!` → `_growend!` (closure) → `unsafe_copyto!` lowers to a
live `ijl_invoke` call:

```llvm
%ret = call ptr @ijl_invoke(
    ptr <fn>,          ; Core.throw_inexacterror (host-VM literal)
    ptr %jlcallframe,  ; alloca [3 x ptr]
    i32 3,
    ptr <mi>)          ; MethodInstance{throw_inexacterror(::Symbol,::Type,::Int64)}
```

(The `throw_inexacterror` comes from Cody-Waite's `check_sign_bit` when
`Int64 → UInt64` for `Csize_t`. It's dead in well-formed code but Julia's
optimiser keeps the call.)

GPUCompiler's IR validator hardcodes `jl_invoke` / `ijl_invoke` rejection,
so we already set `validate = false` in `CompilerConfig` (in
`src/compiler/compilation.jl`) — that gets the IR through validation. But
then the wasm backend can't lower `ptr addrspace(10)` because LLVM-wasm's
default datalayout marks addrspace 10 as `externref` (an opaque host
reference), not a memory pointer. Every `load ptr addrspace(10)` and
`addrspacecast 0 → 10` fails instruction selection.

For M2/M3/M4 this never triggered because every addrspace-10 cast was a
simple alloc-then-use that `InstCombine` folded into addrspace-0 ops.
For our case the cast feeds a live function call, which `InstCombine`
won't cross.

Overriding `GPUCompiler.julia_datalayout` to drop `ni:10:11:12:13` does
**not** help — Julia's frontend emits `ptr addrspace(10)` regardless,
and so does the sub-job.

## The plan: option 2 — wasm function tables + `call_indirect`

Architecture:
1. Maintain a Whisk-side **vtable**: an array of function pointers, one
   slot per unique sub-compiled `MethodInstance`. The vtable lives as an
   LLVM global initialised with constant function pointers.
2. Each sub-compiled function gets a **wasm-compatible wrapper** that
   takes plain `ptr` (addrspace 0) args, addrspacecasts them to 10
   *inside the wrapper body*, and tail-calls the Julia-compiled MI.
   Inside the wrapper, the cast is in a small enough function that
   `AlwaysInliner` + `InstCombine` collapse it before the backend sees it.
3. At each `ijl_invoke` call site, emit a `call_indirect` through the
   wrapper's function pointer (loaded from the vtable). Args extracted
   from the call frame as plain ptrs.
4. `wasm-ld` with `--export-table` exposes the function table; we don't
   need JS to wire anything up.

Why this works where direct calls didn't: the wrapper is the *only*
function with addrspace-10 references, and its body is 5 instructions
(addrspacecasts + a tail call). The `AlwaysInline` heuristic *will*
inline a wrapper of that size into its caller during optimisation.
Once inlined, the caller is the now-non-trivial chain — but the
addrspace-10 ops are dead-code paths surrounded by `unreachable` (since
all our sub-MIs end in trap), and `InstCombine` collapses them.

If that hypothesis fails, the fallback is **option 1** — write an
explicit "strip Julia GC addrspaces" LLVM pass that rewrites
`ptr addrspace(10/11/12/13)` to `ptr addrspace(0)` across the module.
Keep this as a backup; it's mechanical but ~300 lines of careful
LLVM.jl manipulation.

## Concrete steps

### Step 1 — Verify hypothesis on inlining

Before writing 200 lines of pass code:

```julia
# src/compiler/invoke_lowering.jl already exists from the prior session
# (half-working). Modify _replace_invoke_with_direct! to:
#   1. Wrap the sub_fn in a thin trampoline (addrspace-0 args).
#   2. Mark trampoline `alwaysinline`.
#   3. Replace ijl_invoke with a direct call to the trampoline.
# Run AlwaysInlinerPass + InstCombinePass after lower_invokes! in finish_ir!.
# Compile push! test, inspect for any surviving addrspace(10) in the IR.
```

If addrspace(10) is gone after inlining → keep going with this approach,
generalise to `call_indirect` for true dynamic dispatch. If it survives
→ pivot to option 1 (the strip pass).

### Step 2 — Real `call_indirect` lowering

Once direct-call works:

1. Build a vtable global: `@whisk_vtable = constant [N x ptr] [...]` with
   one entry per unique sub-compiled MI.
2. Maintain an `IdDict{Core.MethodInstance, Int}` mapping MI → slot
   index. Persist for the duration of `compile_to_wasm`.
3. Replace `ijl_invoke` with `gep + load + call_indirect`. LLD with
   `--export-table` makes the table addressable.

### Step 3 — `jl_box_*` runtime

Once invokes work, sub-compiled functions may pull in `jl_box_int64`,
`jl_box_int32`, etc. (boxing scalar values into Any-tagged heap objects
inside e.g. exception construction). For each, write a tiny Julia
function and link via the existing `link_compiler_rt!` infrastructure
(model on `whisk_multi3` in `src/runtime/intrinsics.jl`).

### Step 4 — `jl_apply_generic` trap path

True dynamic dispatch needs a method table in wasm linear memory. For
v1, provide a JS-imported stub: `whisk_dynamic_dispatch_unreached` that
calls `js_console(:error, "dynamic dispatch (jl_apply_generic) reached at
runtime — recompile with concrete types")` and traps. Live calls trap
loudly; dead calls get DCE'd.

### Step 5 — diagnostic plumbing for traps

Every wasm trap should emit a `js_console(:error, …)` with at least the
file:line of the offending Julia source. LLVM debug metadata
(`!dbg !N` → `DILocation`) carries file/line. Read it in our trap-rewrite
pass and bake the message as a wasm-side constant string.

## Files to read first when picking this up

1. **`NEXT_PHASE_PLAN.md`** (this file).
2. **`MEMORY.md`** (in `~/.claude/projects/-sim-Programmieren-SimiMaps/memory/`)
   for accumulated user-preference + architectural-decision memories.
3. **`src/compiler/target.jl`** — `WasmCompilerJob`, `finish_module!`,
   `finish_ir!`, intrinsic whitelist.
4. **`src/compiler/compilation.jl`** — `compile_to_wasm`, `link_compiler_rt!`,
   `validate = false`.
5. **`src/compiler/invoke_lowering.jl`** — current half-working state of
   the deferred-compile pass.
6. **`src/runtime/intrinsics.jl`** — i128 helpers (the precedent pattern
   for `jl_box_*`).
7. **`src/runtime/exceptions.jl`** — `js_console`, `unreachable`,
   `whisk_throw_error_inexact`.
8. **`~/.julia/packages/GPUCompiler/Yuvf5/src/driver.jl:213-301`** — the
   `deferred_codegen_jobs` mechanism we're modelling on.
9. **`~/.julia/packages/GPUCompiler/Yuvf5/src/irgen.jl:125-225`** — where
   `lower_throw!` rewrites `jl_throw` calls (worth knowing what runs
   before our passes).
10. **`~/.julia/packages/GPUCompiler/Yuvf5/src/validation.jl:225-265`** —
    the unconditional rejections in `check_ir!` we sidestepped.

## Things NOT to do

- **Do not overlay `Core.throw`.** It's a Julia builtin; `@overlay` is
  silently a no-op for builtins. Inference's builtin path bypasses the
  method table. Verified the wrong way in a prior session by mistaking
  `Base._which(...; method_table=OverlayMethodTable(...))` returning a
  hit for "the override fires". The lookup function is consulted; the
  call dispatcher isn't. Prove end-to-end by inspecting LLVM IR before
  shipping any throw-related @overlay.

- **Do not piecemeal-overlay throw helpers** (`throw_inexacterror`,
  `throw_boundserror`, etc.) as the architectural answer. The user has
  pushed back on this twice. Helper overlays are acceptable as a
  short-term tactic when paired with a documented plan to migrate them
  to a structured pass; not as the long-term design.

- **Do not fall back to running `julia` via Bash.** The `mcp__julia__julia_eval`
  MCP auto-reconnects on the next call after disconnection. Bash julia
  pays full startup + Whisk precompile every invocation and breaks the
  persistent REPL state.

- **Do not write wall-of-IR construction passes.** Runtime helpers
  (`whisk_gc_alloc`, `__multi3`, future `jl_box_*`) must be plain Julia
  functions compiled through the same pipeline as user code, then
  renamed to the C ABI symbol via the `link_compiler_rt!` machinery.
  `src/runtime/gc.jl` is itself a violation of this rule and should be
  refactored eventually — keep it in mind when adding to the GC
  runtime, don't make it worse.

## Verification recipe

After any change:

```julia
# In julia_eval session at /sim/Programmieren/SimiMaps/dev/Whisk
using Pkg
Pkg.test("Whisk")           # must stay 72/72 green

using Whisk
function probe(::Int32)::Int32
    v = Int32[]
    push!(v, Int32(7)); push!(v, Int32(9)); push!(v, Int32(2))
    return @inbounds v[1]
end
out = tempname() * ".wasm"
res = Whisk.compile_to_wasm(probe, Tuple{Int32};
                            out_path = out, name = "probe", optimize = true)
# 1. Should compile without error.
# 2. Inspect imports (use the dump_imports helper from the m5 test).
#    Expected after Step 2 is done: env.gpu_report_exception,
#    env.gpu_signal_exception, env.memmove (and possibly env.js_console).
#    NOT expected: env.ijl_invoke, env.jl_invoke, env.jl_apply_generic.
# 3. Run under Node: should return 7.
```

## Open risks

- **`AlwaysInline` may not fire** on the trampoline if the cost model
  considers it expensive. Mitigation: explicitly call `AlwaysInlinerPass`
  in our `finish_ir!` after `lower_invokes!`.
- **Sub-job's `julia_datalayout` may differ from parent's.** Already
  verified the linker warns about this. May need to set the data layout
  consistently across both before linking.
- **Function-table sizing.** `--initial-memory` already configured;
  `--export-table` and an initial table size need to be added to the
  `lld` invocation in `link_wasm` if Step 2 happens.
- **`uses_julia_runtime`** is currently default-false. Setting it true
  changes a lot (skips `lower_throw!`, links libjulia symbols). Probably
  keep false; revisit only if invoke-lowering needs it.
