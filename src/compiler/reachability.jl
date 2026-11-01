## Phase 1 — closed-world reachability (ARCHITECTURE.md).
##
## Walk the typed call graph from an entry point and produce the
## program's closed world:
##
##   * `methods`   — every MethodInstance reachable through static
##                   edges (:invoke and resolvable :call).
##   * `dynsites`  — call sites inference could NOT devirtualize, with
##                   their argtype lattice elements. Phase 5 builds
##                   dispatch tables for these.
##   * `universe`  — every concrete DataType the program can
##                   instantiate or reference. The candidate pool for
##                   dynamic dispatch and (later) the data image's
##                   descriptor set.
##
## The walk uses inferred CodeInfo from `Base.code_typed_by_type` at
## the current world. NOTE: reachability currently consults the NATIVE
## method table; Whisk's overlay (quirks.jl) only swaps throw helpers,
## which never affects which user methods are reachable. If overlays
## ever change dispatch targets, switch retrieval to the Whisk
## interpreter.

struct DynamicSite
    mi::Core.MethodInstance      # enclosing method instance
    argtypes::Vector{Any}        # lattice elements at the call (incl. callee)
    line::Int32                  # statement index (diagnostics)
end

struct ReachableWorld
    methods::Vector{Core.MethodInstance}
    dynsites::Vector{DynamicSite}
    universe::Vector{DataType}   # concrete types, deduped
end

"""
    reachable_world(f, argtypes::Type{<:Tuple}) → ReachableWorld

Compute the closed world for entry `f(::argtypes...)`. Bounded by
`max_methods` MethodInstances as a runaway guard (a diagnostic error,
not a silent cap).
"""
function reachable_world(@nospecialize(f), @nospecialize(argtypes::Type{<:Tuple});
                          max_methods::Int = 10_000)
    seen_mi  = Set{Core.MethodInstance}()
    worklist = Core.MethodInstance[methodinstance(typeof(f), argtypes)]
    dynsites = DynamicSite[]
    universe = Set{DataType}()
    expanded_at = Dict{Int, Int}()   # dynsite index → universe size at last expansion
    percallee_cache = Dict{Tuple{DataType, Int},
                           Union{Nothing, Core.MethodInstance}}()

    register_type!(universe, typeof(f))
    for p in argtypes.parameters
        p isa DataType && register_type!(universe, p)
    end

    while true
        while !isempty(worklist)
            mi = pop!(worklist)
            mi in seen_mi && continue
            push!(seen_mi, mi)
            length(seen_mi) > max_methods &&
                error("Whisk reachability: method universe exceeded $(max_methods) — runaway expansion; raise max_methods if the program is genuinely this large")

            ci = inferred_code(mi)
            ci === nothing && continue
            scan_code!(worklist, dynsites, universe, mi, ci)
        end
        # Dynamic sites' candidate methods belong to the closed world
        # too — phase 5 sub-compiles them, so THEIR bodies' dynamic
        # sites must carry argtype provenance as well. Expand every
        # site against the current universe and keep walking until
        # nothing new appears (the universe is finite and only grows,
        # so this converges; `seen_mi` makes re-expansion idempotent).
        # A site's candidate set can only change when the universe has
        # grown since it was last expanded — skip the rest (the sweep
        # itself is method-table-lookup heavy).
        added = false
        usize = length(universe)
        for (si, site) in enumerate(dynsites)
            get(expanded_at, si, -1) == usize && continue
            expanded_at[si] = usize
            for mi in site_candidates(universe, site; percallee_cache)
                mi in seen_mi && continue
                push!(worklist, mi)
                added = true
            end
        end
        added || break
    end

    return ReachableWorld(collect(seen_mi), dynsites,
                          sort!(collect(universe); by = string))
end

# Retrieve inferred, optimized CodeInfo for `mi` at the current world.
function inferred_code(mi::Core.MethodInstance)
    try
        results = Base.code_typed_by_type(mi.specTypes; optimize = true)
        isempty(results) && return nothing
        return results[1][1]::Core.CodeInfo
    catch
        return nothing
    end
end

function register_type!(universe::Set{DataType}, @nospecialize(T))
    T isa DataType || return
    isconcretetype(T) || return
    T in universe && return
    push!(universe, T)
    # Field types are storable, hence reachable as values.
    if !(T <: Memory) && !isprimitivetype(T)
        for i in 1:fieldcount(T)
            ft = fieldtype(T, i)
            ft isa DataType && isconcretetype(ft) && register_type!(universe, ft)
        end
    end
    return
end

# Lattice element → argtype for dispatch candidate matching.
widen_arg(@nospecialize(t)) = Core.Compiler.widenconst(t)

function scan_code!(worklist, dynsites, universe, mi::Core.MethodInstance,
                     ci::Core.CodeInfo)
    ssatypes = ci.ssavaluetypes isa Vector ? ci.ssavaluetypes : Any[]

    argtype_of = function (@nospecialize(x),)
        if x isa Core.SSAValue
            return x.id <= length(ssatypes) ? widen_arg(ssatypes[x.id]) : Any
        elseif x isa Core.Argument
            sig = mi.specTypes
            return x.n <= length(sig.parameters) ? sig.parameters[x.n] : Any
        elseif x isa GlobalRef
            return isdefined(x.mod, x.name) ? typeof(getfield(x.mod, x.name)) : Any
        elseif x isa QuoteNode
            return typeof(x.value)
        else
            return typeof(x)
        end
    end

    for (idx, stmt) in enumerate(ci.code)
        if stmt isa Expr
            if stmt.head === :invoke
                target = stmt.args[1]
                target_mi = target isa Core.MethodInstance ? target :
                            target isa Core.CodeInstance ? target.def : nothing
                target_mi isa Core.MethodInstance && push!(worklist, target_mi)
                for a in stmt.args[3:end]
                    t = argtype_of(a)
                    t isa DataType && register_type!(universe, t)
                end
            elseif stmt.head === :call
                callee = stmt.args[1]
                ats = Any[argtype_of(a) for a in stmt.args]
                for t in ats
                    t isa DataType && register_type!(universe, t)
                end
                callee_t = ats[1]
                if callee_t isa DataType && callee_t <: Core.Builtin
                    # Builtins never add method edges. `Core._apply_iterate`
                    # wraps a generic callee — treat as dynamic on its target.
                    continue
                end
                if all(t -> t isa DataType && isconcretetype(t), ats)
                    matched = Base._which(Tuple{ats...}; raise = false)
                    if matched !== nothing
                        push!(worklist,
                              Core.Compiler.specialize_method(matched))
                        continue
                    end
                end
                push!(dynsites, DynamicSite(mi, ats, Int32(idx)))
            elseif stmt.head === :new
                t = stmt.args[1]
                t = t isa GlobalRef ? getfield(t.mod, t.name) : t
                t isa DataType && register_type!(universe, t)
            elseif stmt.head === :foreigncall
                # No method edges; runtime symbols handled by the
                # lowering passes / validator.
            end
        end
    end
    return
end

"""
    slot_constraint(m::Core.MethodMatch, k) → Type | nothing

Declared type of positional slot `k` (1 = the callee) of `m`'s method
signature, with `Vararg` unwrapped, bare typevars widened to their
upper bound, and remaining free typevars closed over the method's
`where` environment — i.e. a sound over-approximation of what slot `k`
accepts, safe to `typeintersect` against. Returns `nothing` when the
method has no slot `k`.
"""
function slot_constraint(m::Core.MethodMatch, k::Int)
    usig = Base.unwrap_unionall(m.method.sig)
    usig isa DataType || return nothing
    ps = usig.parameters
    p = if k <= length(ps)
        ps[k]
    elseif !isempty(ps) && Core.Compiler.isvarargtype(ps[end])
        ps[end]
    else
        return nothing
    end
    Core.Compiler.isvarargtype(p) && (p = Base.unwrapva(p))
    while p isa TypeVar
        p = p.ub
    end
    p isa Type || return nothing
    Core.Compiler.has_free_typevars(p) ?
        Base.rewrap_unionall(p, m.method.sig) : p
end

"""
    slot_feasible(universe, ms, k, hint) → Vector{DataType}

Universe types that can occupy argument slot `k` (1 = the callee) of a
call to one of the methods `ms`: the type must intersect slot `k` of
SOME method AND the inference `hint` for that slot (`Any` = no
provenance constraint). This pruning is what keeps closed-world
dispatch tables per-site small — without it every Number in the
universe is a candidate at every `+` site.
"""
function slot_feasible(universe, ms::Vector, k::Int, @nospecialize(hint))
    cons = unique!(Any[c for c in (slot_constraint(m, k) for m in ms)
                       if c !== nothing])
    out = DataType[]
    for U in universe
        (hint === Any || typeintersect(U, hint) !== Union{}) || continue
        any(p -> typeintersect(U, p) !== Union{}, cons) && push!(out, U)
    end
    out
end

"""
    arg_keyed_candidates(fk_type, nargs, universe; hints, fixed, budget)
        → [(U, payload)…]

Closed-world dispatch candidates for a known-singleton callee, keyed
by the FIRST RUNTIME argument's type: payload is the MethodInstance
when the slot-`U` signature names a unique usable method, or
`[(V, mi)…]` pairs (second-runtime-slot refinement, emitted as a
nested tag switch) when it does not (`+(x, v)` — every `+` method
constrains BOTH operands).

`fixed` (partial devirtualization) carries the frame slots that
decoded to host constants — `missing` entries are runtime. Constant
slots contribute their exact type to every probe signature
(`convert(Float64, v)` probes `Tuple{typeof(convert), Type{Float64},
V}`); only runtime slots are swept against the universe.

`hints` are the inferred argument types at the call site(s) — dynsite
provenance — and bound each runtime slot's universe sweep. Without
hints the table covers everything the callee's signatures admit, which
explodes combinatorially for method-rich generics; `budget` caps the
number of scheduled sub-compiles, returning an empty table (→ loud
runtime trap) with a compile-time diagnostic instead of compiling for
minutes.
"""
function arg_keyed_candidates(@nospecialize(fk_type), nargs::Int, universe;
                              hints::Union{Nothing, Vector{Any}} = nothing,
                              fixed::Union{Nothing, Vector{Any}} = nothing,
                              budget::Int = 256)
    out = Tuple{DataType, Any}[]
    ms = Base._methods_by_ftype(Tuple{fk_type, Vararg{Any}}, -1,
                                Base.get_world_counter())
    ms isa Vector || return out
    # Keep methods that can accept `nargs` call arguments (method
    # signatures count the callee; vararg counts as one slot).
    filter!(m -> m.method.isva ? nargs + 1 >= m.method.nargs - 1 :
                                 nargs + 1 == m.method.nargs, ms)
    isempty(ms) && return out
    hint(k) = hints === nothing ? Any : hints[k - 1]
    fixedtype(i) = fixed === nothing || fixed[i] === missing ? nothing :
                   fixed[i] isa Type ? Type{fixed[i]} : typeof(fixed[i])
    rslots = fixed === nothing ? collect(1:nargs) :
             [i for i in 1:nargs if fixed[i] === missing]
    isempty(rslots) && return out   # all-constant sites devirtualize instead
    r1 = rslots[1]
    r2 = length(rslots) >= 2 ? rslots[2] : 0
    # Probe-signature slot for arg `i`: the swept type at the runtime
    # key slots, the decoded constant's type at fixed slots, Any
    # elsewhere (3rd+ runtime slot — the table stays sound, those args
    # just stay boxed).
    slotat = function (i, @nospecialize(U), @nospecialize(V))
        i == r1 && return U
        (i == r2 && V !== nothing) && return V
        ft = fixedtype(i)
        ft === nothing ? Any : ft
    end
    # A match is usable when its non-callee params are `Any` (arg
    # stays boxed), concrete, or a constant `Type{X}` (a ghost — fixed
    # slots produce these; `isconcretetype(Type{Float64})` is false).
    usable(m) = m !== nothing && m.spec_types isa DataType &&
        all(p -> p === Any || (p isa DataType && (isconcretetype(p) ||
                Core.Compiler.isconstType(p))),
            m.spec_types.parameters[2:end])
    nsub = 0
    second = r2 != 0 ? slot_feasible(universe, ms, r2 + 1, hint(r2 + 1)) :
             DataType[]
    for U in slot_feasible(universe, ms, r1 + 1, hint(r1 + 1))
        sig = Tuple{fk_type, (slotat(i, U, nothing) for i in 1:nargs)...}
        matched = Base._which(sig; raise = false)
        if usable(matched)
            push!(out, (U, Core.Compiler.specialize_method(matched)))
            nsub += 1
        elseif r2 != 0
            refined = Tuple{DataType, Core.MethodInstance}[]
            for V in second
                sig2 = Tuple{fk_type, (slotat(i, U, V) for i in 1:nargs)...}
                m2 = Base._which(sig2; raise = false)
                usable(m2) || continue
                push!(refined, (V, Core.Compiler.specialize_method(m2)))
                nsub += 1
            end
            isempty(refined) || push!(out, (U, refined))
        end
        if nsub > budget
            @warn "Whisk closed-world dispatch: callee $(fk_type) at arity " *
                  "$(nargs) exceeds the dispatch-table budget ($(budget) " *
                  "sub-compiles)$(hints === nothing ? " — site has no dynsite provenance" : "")" *
                  "; the site will trap at runtime"
            return empty!(out)
        end
    end
    out
end

"""
    dynsite_hints(world, fk_type, nargs) → Vector{Any} | nothing

Union of the inferred argument types over every dynamic site whose
callee lattice element can be `fk_type` at arity `nargs` — the
provenance bound for a phase-5 dispatch table. `nothing` when no
walked site matches (code only reachable through lowering-time
sub-compiles outside the walked world, e.g. runtime helpers): the
caller must then fall back to an unhinted, budget-capped table.

Per-(callee, arity) union rather than per-site precision: the LLVM
call site does not yet carry which MethodInstance it came from.
DEBT(provenance): thread a function-name → MethodInstance map through
codegen so each site uses exactly its own dynsite's argtypes.
"""
function dynsite_hints(world::ReachableWorld, @nospecialize(fk_type), nargs::Int)
    hints = nothing
    for site in world.dynsites
        length(site.argtypes) == nargs + 1 || continue
        # EXACT callee match only. An abstract callee element (`Any` —
        # `f(val)` over stored listeners) would intersect every
        # `fk_type` and union its untyped args into every table,
        # erasing the provenance this function exists to provide;
        # those sites dispatch callee-keyed instead.
        site.argtypes[1] === fk_type || continue
        if hints === nothing
            hints = Any[t isa Type ? t : Any for t in site.argtypes[2:end]]
        else
            for k in 1:nargs
                t = site.argtypes[k + 1]
                hints[k] = Union{hints[k], t isa Type ? t : Any}
            end
        end
    end
    return hints
end

"""
    arity_unique_candidate(T, nargs) → MethodInstance | nothing

The unique arity-`nargs` call method of callee type `T`, specialized
at its own signature — the candidate phase-5 lowering would emit for a
`T`-tagged callee. `nothing` when `T` has zero or several
arity-matching methods (several would need argtype refinement).
"""
function arity_unique_candidate(@nospecialize(T), nargs::Int)
    matches = Base._methods_by_ftype(Tuple{T, Vararg{Any}}, -1,
                                     Base.get_world_counter())
    matches isa Vector || return nothing
    found = nothing
    for m in matches
        m.spec_types isa DataType || continue
        ps = m.spec_types.parameters
        length(ps) == nargs + 1 || continue
        (ps[1] isa DataType && isconcretetype(ps[1])) || continue
        all(p -> p === Any || (p isa DataType && isconcretetype(p)),
            ps[2:end]) || continue
        found === nothing || return nothing   # ambiguous: >1 arity match
        found = Core.Compiler.specialize_method(m)
    end
    return found
end

"""
    site_candidates(universe, site; percallee_cache) → Vector{Core.MethodInstance}

Closed-world candidate set for a dynamic call site, bounded by the
site's inferred argtypes. Singleton-function and Type callees go
through the slot-pruned `arg_keyed_candidates`; dynamic callees
(`f(val)` over stored listeners) enumerate the universe's Function
types that intersect the callee lattice element and have a unique
arity-matching method — mirroring what phase-5 lowering sub-compiles,
so the walk sees the same bodies the artifact will contain.
"""
function site_candidates(universe, site::DynamicSite;
                         percallee_cache::Union{Nothing, Dict{Tuple{DataType, Int},
                             Union{Nothing, Core.MethodInstance}}} = nothing)
    out = Core.MethodInstance[]
    ats = site.argtypes
    nargs = length(ats) - 1
    ct = ats[1]
    if ct isa DataType && isconcretetype(ct) &&
       (Base.issingletontype(ct) || ct <: Type) && nargs >= 1
        hints = Any[t isa Type ? t : Any for t in ats[2:end]]
        for (_, payload) in arg_keyed_candidates(ct, nargs, universe; hints)
            if payload isa Core.MethodInstance
                push!(out, payload)
            else
                for (_, mi) in payload
                    push!(out, mi)
                end
            end
        end
        return out
    end
    for T in universe
        T <: Function || continue
        T <: Core.Builtin && continue
        T === Core.IntrinsicFunction && continue
        (ct === Any || (ct isa Type && typeintersect(T, ct) !== Union{})) ||
            continue
        # The per-(T, nargs) enumeration is site-independent and
        # method-table heavy — memoize across sites and rounds.
        cand = percallee_cache === nothing ?
            arity_unique_candidate(T, nargs) :
            get!(() -> arity_unique_candidate(T, nargs),
                 percallee_cache, (T, nargs))
        cand === nothing || push!(out, cand)
    end
    return out
end
