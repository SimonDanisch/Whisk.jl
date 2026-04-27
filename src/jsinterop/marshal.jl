## JS marshalling — M5 territory.
##
## Once linear memory + allocator land, exporting a Julia function to JS
## means: copy input typed arrays into wasm memory, call wasm, copy
## result out. Ergonomic helpers go here.
