## IR validation
##
## M7 territory — friendly errors for unsupported ccalls. Stub for now.

"""
    UnsupportedCcallError

Thrown at IR-validation time when Julia source uses a ccall that Whisk
doesn't translate to a wasm equivalent. Carries the ccall symbol and
the source location.
"""
struct UnsupportedCcallError <: Exception
    symbol::Symbol
    location::String
end

function Base.showerror(io::IO, e::UnsupportedCcallError)
    print(io, "Whisk.UnsupportedCcallError: ccall to `:", e.symbol,
              "` at ", e.location, " — ccall is the only invalid Julia ",
              "code in Whisk; rewrite as native Julia or add a translation.")
end
