## Whisk method-table overrides
##
## Mirrors CUDA's `@device_override` / Abacus's `@abacus_override`.
## Used sparingly — most Julia stdlib should compile unchanged. The few
## genuine wasm-incompatibilities (e.g., system I/O, OS time) get
## redirected here.

using Base.Experimental: @overlay, @MethodTable
@MethodTable method_table

# Empty for M1. First overrides land at M3 (allocator / GC) and M4
# (anything that surfaces in Dict / Set tests).
