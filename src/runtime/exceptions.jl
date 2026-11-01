## Exception handling
##
## Wasm has no native exception machinery. Whisk's model:
##
##  1. Most `throw(ex)` calls go through GPUCompiler's existing handling
##     (`gpu_report_exception(msg_ptr) + llvm.trap`). JS sees a
##     `WebAssembly.RuntimeError` with the message GPUCompiler bakes in
##     as a `@exception.N` global. Good enough for `BoundsError`,
##     `ArgumentError`, `DomainError` etc. with concrete construction.
##
##  2. Some Base helpers construct exceptions whose type is itself an
##     argument boxed to `Any` (e.g. `throw_inexacterror(:convert, To, x)`
##     stores `To::Any` in `InexactError.T::Any`). That boxing produces
##     a dynamic dispatch GPUCompiler's IR validator rejects. We overlay
##     these specific helpers (in `quirks.jl`) and route them to
##     `whisk_throw_error(...)` here, which skips exception-object
##     construction entirely.
##
##  3. Each `whisk_throw_error` method is plain Julia: it can introspect
##     the relevant arguments, format whatever's useful, then call
##     `js_console(:error, msg)` and trap via `unreachable()`.

"""
    js_console(level::Symbol, msg::String)

Emit a console message via the host JS shim. `level` is `:log`,
`:warn`, or `:error`; anything else falls through to `:log`. The
message bytes are passed as `(ptr, len)` into wasm linear memory and
JS decodes UTF-8 + dispatches to `console.log` / `.warn` / `.error`.
"""
@inline function js_console(level::Symbol, msg::String)
    lvl = level === :error ? Int32(2) :
          level === :warn  ? Int32(1) :
                             Int32(0)
    GC.@preserve msg ccall(:js_console, Cvoid,
                           (Int32, Ptr{UInt8}, UInt64),
                           lvl, pointer(msg), UInt64(sizeof(msg)))
end

# LLVM `unreachable` instruction — the block terminator we use after
# a `js_console` call so the basic block is well-formed and any code
# downstream of the throw is pruned.
@inline function unreachable()
    Base.llvmcall("unreachable", Nothing, Tuple{})
end

# ── Throw entry points (called from quirks.jl overlays) ──────────────
# Each is a regular Julia function. Bodies stay light — heavy
# introspection / formatting would drag IO/show into every wasm.

@noinline function whisk_throw_error_inexact(func::Symbol)
    msg = func === :trunc ? "InexactError: trunc" : "InexactError: convert"
    js_console(:error, msg)
    unreachable()
end
