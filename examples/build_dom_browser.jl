## Build the wasm asset used by `dom_browser.html`.
##
## Usage:
##     julia --project=. examples/build_dom_browser.jl

using Whisk

const DOM_ID_ANSWER = Int32(1)

function dom_get_answer_node()::UInt64
    return ccall(:dom_get_element_by_id, UInt64, (Int32,), DOM_ID_ANSWER)
end

function dom_set_text_i32(node::UInt64, value::Int32)::Nothing
    ccall(:dom_set_text_i32, Cvoid, (UInt64, Int32), node, value)
    return nothing
end

function dom_score(n::Int32)::Int32
    total = Int32(0)
    for i in Int32(1):n
        total += i * i
    end
    return total
end

function render_dom_score(n::Int32)::Int32
    result = dom_score(n)
    node = dom_get_answer_node()
    dom_set_text_i32(node, result)
    return result
end

out_path = joinpath(@__DIR__, "dom_browser.wasm")
compiled = Whisk.compile_to_wasm(render_dom_score, Tuple{Int32};
                                 out_path,
                                 name = "render_dom_score",
                                 optimize = true)

println("wrote : ", compiled.wasm_path)
println("entry : ", compiled.entry_name)
println("size  : ", filesize(compiled.wasm_path), " bytes")
