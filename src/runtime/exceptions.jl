## Exception handling — stubs for M1.
##
## Wasm has its own `exnref`/`throw`/`catch` proposal (Phase 4 as of LLVM
## 18). For M1 we sidestep entirely: error paths reduce to `unreachable`,
## so calling them traps. Real exceptions in M3+.
