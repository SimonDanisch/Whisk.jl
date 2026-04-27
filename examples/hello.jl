## M1: smallest possible function — no allocator, no GC, just integer
## arithmetic. Compiles to wasm, runs in V8.

using Whisk

# The function we'd like to ship.
inc(x::Int32) = x + Int32(1)

result = Whisk.compile_to_wasm(inc, Tuple{Int32};
                               out_path = "/tmp/hello.wasm",
                               name     = "inc",
                               optimize = true)

println("wasm  : ", result.wasm_path, "  (", filesize(result.wasm_path), " bytes)")
println("entry : ", result.entry_name)
println()
println("--- LLVM IR ---")
println(result.llvm_ir)

# Smoke-test in Node — separate script will execute.
