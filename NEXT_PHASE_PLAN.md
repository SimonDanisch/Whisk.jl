# Whisk · Next Phase Plan — Unified Type Descriptors

Self-contained plan for a fresh session. Mission context: compile Makie's
scene-construction pipeline to wasm so WGLMakie becomes a standalone
plotly-like JS library (see `~/.claude/projects/-sim-Programmieren-WhiskDev/`
memory: `whisk-wglmakie-mission`). The Makie-ladder corpus at
`/sim/Programmieren/WhiskDev/ladder/run.jl` drives priorities.

## Where Whisk is right now (2026-06-11)

`Pkg.test("Whisk")` **153/153** green; ladder **10/10** green
(GeometryBasics Vec/Mat math, Vector{Point2f}, Colors/N0f8 conversion,
colormap interpolation, push!+bbox, Dict{Symbol,Float32}, Vector{Any}
with boxed scalars, Union{Nothing,Float32}).

Recent landings (same day):
- `libraries = false` — GPUCompiler's rtlib no longer linked; Whisk owns
  the full runtime (`ijl_box_*` for 12 scalar types via
  `COMPILER_RT_HELPERS`).
- Interior host-pointer reconstruction (`rewrite_host_pointer_constants!`):
  probes deltas 0/8/16, two-level type-tag validation (`mincore`-guarded),
  layout-faithful globals for String / Symbol / Memory{bits} /
  Vector{bits} / boxed-isbits literals / DataType descriptors.
- Type-tag v1: `_make_wasm_datatype_global!` descriptors ({objectid,
  name}, 16-aligned, named `_whisk_type_<objectid hex>`); box helpers
  fetch tags via `whisk_typetag_<T>` externs (`define_box_typetags!`);
  isa/typeassert works on box values.
- Exception lowering v1 (`lower_exception_intrinsics!`):
  `julia.except_enter` → `{0, null}`, `ijl_excstack_state` → 0,
  `ijl_pop_handler*` erased. try/finally + try/catch compile; the
  non-throwing path runs; throws inside try still trap loudly.

## DONE (later same day): unified type descriptors LANDED

The plan below was executed in full — `whisk_type_descriptor!` unified globals,
per-type keying, `ijl_get_nth_field_checked` runtime helper, small-typetag
mapping (tags are BYTE offsets into jl_small_typeof, idx<<3), tag-prefixed
materialized literals (object base = const-GEP +8), typed String constructors,
`--global-base=65536`. Suite 153/153, ladder 10/10 (incl. string building and
Dict{Symbol}). Remaining follow-ups tracked as tasks: exceptions v2 (real
unwinding), non-bits Memory materialization (Memory{Symbol}/Memory{Any}),
then DYNAMIC DISPATCH (descriptors gain a method-table slot).

## The blocker WAS: two parallel tag systems (historical)

`object[-8]` means two different things today:

1. Heap objects from `whisk_gc_alloc_typed` (16-byte GC header:
   `{u64 size, u64 layout|flags}` ending at the payload): `obj[-8]` is
   the **GC layout descriptor** (`runtime/layout.jl` wire format
   `{u64 elsz, u64 nptrs, u64 offsets[]}`), deduped **by shape** across
   types, low 3 bits used for mark/flags.
2. Box values from `ijl_box_*` and materialized host literals:
   `obj[-8]` is the **DataType descriptor** (`_whisk_type_*`).

isa/typeassert/typeof compare `obj[-8] & ~15` against DataType
descriptors — correct for (2), wrong for (1). `ijl_get_nth_field_checked`
(ladder R10 blocker, `print_to_string` vararg iteration) needs per-type
field metadata that neither record has.

## The plan: one descriptor global per type

Merge both records into a single per-type global. The GC mark walker
reads a prefix; type identity is the global's address; field metadata
appended for runtime field access:

```
UnifiedDescriptor (16-byte aligned, one per DataType):
  u64 elsz              // unchanged layout prefix (mark walker)
  u64 nptrs
  u64 offsets[nptrs]
  -- appended type metadata (walker never reads past offsets) --
  u64 objectid          // == objectid(T) on host
  u64 payload_size      // aligned size for re-boxing fields
  u64 nfields
  u64 fields_off        // byte offset of fields records, 0 if none
  name bytes (NUL)
  fields records: nfields × { u64 offset, u64 desc_addr, u64 kind }
                  // kind: 0 = pointer field, 1 = inline bits,
                  //       2 = unsupported (helper traps with msg)
```

Steps:
1. **Key layout descriptors by type, not shape** (`runtime/layout.jl`):
   one global per DataType reaching `gc_alloc_obj` /
   `alloc_genericmemory`. Accept the rodata growth.
2. **Extend the emitter** with the metadata fields above. Two-phase
   construction (create global → build initializer) so field-descriptor
   references can recurse/cycle.
3. **Converge all paths** on the unified emitter:
   `_make_wasm_datatype_global!`, `lower_gc_alloc_wasm!`,
   `define_box_typetags!`, `emit_layout_addr_table!` (keep JS manifest
   tag ids stable — they index the table, not the format).
4. **Alignment/flags audit**: descriptors 16-aligned so typeassert's
   `& ~15` and the walker's low-3-bit flags coexist.
5. **`whisk_jl_get_nth_field_checked(v, i)`** as a plain-Julia
   runtime helper (COMPILER_RT_HELPERS): read `v[-8] & ~15` → descriptor
   → bounds-check i → kind 0: load ptr; kind 1: `_whisk_alloc_box`
   (tag = field desc) + memcpy payload_size; kind 2: loud trap.
6. Ladder R10 goes green; add R11 (typeassert on a heap-allocated
   tuple in Vector{Any} — exercises unification directly), R12
   (`hash(::Symbol)` + Dict{Symbol,Any}).

## ACTIVE: closed-world dispatch (task #3, ladder R13)

R13 shape: `obs.listener::Any` holds a closure; `(f::Function)(x)` →
`jl_apply_generic`. Implementation plan (ARCHITECTURE.md phases 1+5):

1. `src/compiler/reachability.jl` — `reachable_world(f, tt)`:
   worklist over inferred code (same interpreter family GPUCompiler
   uses). For each MI: retrieve inferred CodeInfo (code_typed with
   the right interpreter/world); collect
   (a) :invoke edges → worklist,
   (b) statically-resolvable :call edges → worklist,
   (c) unresolved :call sites → dynamic-site table (argtype lattice),
   (d) every concrete type instantiated/referenced → type universe.
2. Dispatch table build: for each dynamic site with signature
   `(Any, argtypes...)`: candidates = {T in universe : T <: Function,
   hasmethod(T.instance-ish, argtypes)} (callable structs later). Each
   candidate → methodinstance → sub-compile via existing
   `_sub_compile_mi!` machinery.
3. IR lowering `lower_apply_generic!(mod, world)`: at each
   `ijl_apply_generic(%f, %frame, nargs)` site: tag = `%f[-8] & ~15`,
   objectid = `load(tag+16)`, `switch` over candidate
   objectid(closure-type) literals → per-case direct call to the
   sub-compiled MI (args from frame via `_load_arg!`, closure object
   itself as arg 1); default → loud trap listing the considered
   candidates. Replaces `lower_dynamic_dispatch!`'s unconditional trap
   when the site is resolvable.
4. Closure type descriptors: `whisk_type_descriptor!` already handles
   any concrete struct — closures are plain structs; their alloc sites
   are typed so runtime tags exist.
5. R13 green; new rungs: Vector{Function} multi-listener, callable
   struct, two-candidate site.

## After that (in order)

- **Dynamic dispatch** (jl_apply_generic → vtable + call_indirect):
  descriptors gain a method-table slot; `lower_invokes!` already
  sub-compiles MIs — generalize to a (f, argtags) → wasm-func table.
- **Exceptions v2**: real unwinding (wasm exnref EH or error-code
  threading) so catch actually catches — Makie uses try/catch in
  font/attribute fallbacks.
- **Makie rungs**: Observables-lite, MsgPack decode of a PlotSpec dict,
  convert_arguments(Scatter, …), colormap pipeline end-to-end.

## Things NOT to do (carried over + new)

- Do not overlay `Core.throw` / builtins (silent no-op).
- Do not run julia via Bash for iteration — use the julia MCP
  (`env_path` = /sim/Programmieren/WhiskDev). **The MCP session has NO
  Revise** despite its docs: `julia_restart` after every Whisk edit.
- Runtime helpers must be plain Julia via `link_compiler_rt!`. Use
  `ccall("extern <sym>", llvmcall, …)` for runtime→runtime calls — the
  `Base.llvmcall((decl, body), …)` tuple form silently compiles to a
  trap under the GPUCompiler interpreter (does not const-fold).
- Do not dedup type descriptors by layout shape — identity is the point.

## Verification recipe

```julia
# julia MCP, env_path=/sim/Programmieren/WhiskDev
using Pkg; Pkg.test("Whisk")               # must stay 153/153
include("/sim/Programmieren/WhiskDev/ladder/run.jl")  # target: 10/10 + new rungs
```
