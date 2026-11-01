## Reflection helpers

"""
    @code_wasm f(args...)

Show the wasm text (`.wat`) for a compiled Julia call. Convenience
wrapper around `compile_to_wasm` + `wasm-dis`. Stub for now — fills in
once M1 is green.
"""
macro code_wasm(call)
    error("@code_wasm not yet implemented (M1 in progress)")
end
