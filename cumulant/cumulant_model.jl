using QuantumCumulants
using ModelingToolkit
using Serialization
using Printf

"""
Model definition for the cumulant-expansion comparison (QuantumCumulants.jl,
indexed emitter operators, scaled for identical emitters), plus a disk cache
for the symbolic derivation.

The derivation is the bottleneck of the whole method: `complete` closes the
hierarchy by factorizing every order-(M+1) average that appears on a
right-hand side into products of cumulants of order <= M, a sum over set
partitions whose term count grows like the Bell number B(M+1) (52 at M=4,
21147 at M=8, 4.7e14 at M=20).  Measured on this project's model:

    order 4: 18 equations, 27 s
    order 5: 28 equations, 20 s
    order 6: 48 equations, 123 s

i.e. the equation count grows mildly (the ODE stays tiny) while the derivation
cost grows by roughly a factor 6 per order, putting the practical ceiling near
M = 8-9.  Since the derived system is independent of *all* physical parameters
-- n0 enters only as the symbolic parameter N after `scale` -- it is derived
once per order and reused for every pump rate, emitter number, and dephasing
value.  That is what this cache is for.

Note on low orders: QuantumCumulants cannot form an equation of motion for
<a'a'aa> below order 4 ("Cannot form cumulant expansion of derivative"), so
orders 2 and 3 track only <a'a> here.  This is not a limitation of the solver
but of the observable: by the U(1) phase invariance below, <a> = <aa> = 0, so
the Gaussian factorization gives <a'a'aa> = 2<a'a>^2 identically and hence
g2(0) = 2 exactly at any order <= 3, for any parameters.  See
`g2_below_fourth_order`.
"""

# Bump when the physical model (H, jumps, or rates) changes, so that stale
# caches from an earlier model are rejected rather than silently reused.
const CUMULANT_MODEL_TAG = "tavis-cummings-indexed-local-A-P-D-v1"

# --- Model: cavity + N identical two-level atoms (indexed/scaled) ---

const hc = FockSpace(:cavity)
const ha = NLevelSpace(:atom, (:g, :e))
const h = hc ⊗ ha

@qnumbers a::Destroy(h)
σ(α, β, i) = IndexedOperator(Transition(h, :σ, α, β), i)
σn(α, β, k::Integer) = IndexedOperator(Transition(h, :σ, α, β), k)

@cnumbers N g kappa gamma_A P gamma_D

const i_idx = Index(h, :i, N, ha)

# H = g * sum_i (a' sigma_i^{ge} + a sigma_i^{eg}); sigma^{ge} = |g><e| is the
# local lowering operator, as in the paper.
const H_CUMULANT = g * Σ(a' * σ(:g, :e, i_idx) + a * σ(:e, :g, i_idx), i_idx)

# Cavity loss kappa (= gamma_c); local decay gamma_A; local incoherent pump P
# (= gamma_P); local pure dephasing gamma_D on sigma^{ee} (the paper's
# convention, no factor-of-4 conversion needed).
const J_CUMULANT = [a, σ(:g, :e, i_idx), σ(:e, :g, i_idx), σ(:e, :e, i_idx)]
const RATES_CUMULANT = [kappa, gamma_A, P, gamma_D]
const CUMULANT_PARAMETERS = [N, g, kappa, gamma_A, P, gamma_D]

# Phase-invariance filter for the U(1) symmetry of this Liouvillian.  This is
# the moment-level counterpart of the conserved charge c of the paper (App. A):
# discarding non-invariant moments is EXACT, not an approximation, because the
# initial state (vacuum + all atoms in |g>) is itself phase invariant, so the
# discarded moments vanish identically for all time.
φ(x::Average) = φ(x.arguments[1])
φ(::Destroy) = -1
φ(::Create) = 1
φ(x::QuantumCumulants.QMul) = sum(map(φ, x.args_nc))
function φ(t::Transition)
    t.i == :e && t.j == :g && return 1
    t.i == :g && t.j == :e && return -1
    return 0
end
φ(x::IndexedOperator) = φ(x.op)
φ(x::SingleSum) = φ(x.term)
φ(x::QuantumCumulants.AvgSums) = φ(arguments(x))
phase_invariant(x) = iszero(φ(x))

"""Is `order` high enough for <a'a'aa> to be a dynamical variable?"""
g2_capable(order::Integer) = order >= 4

"""Observables to derive equations for at a given truncation order."""
target_operators(order::Integer) =
    g2_capable(order) ? [a' * a, a' * a' * a * a] : [a' * a]

"""
g2(0) at truncation order <= 3, where it is fixed analytically rather than
solved for.  Phase invariance forces <a> = <aa> = 0, so the only surviving
Gaussian pairings of <a'a'aa> are the two <a'a><a'a> terms, and the
third-order cumulants enter only multiplied by <a> = 0.
"""
g2_below_fourth_order(::Integer) = 2.0

# --- Symbolic derivation with a disk cache ---

# What is serialized is a tree of SymbolicUtils/Symbolics terms, so the cache is
# only valid for the versions of the whole symbolic stack that wrote it -- NOT
# just QuantumCumulants.  Stamping on QuantumCumulants alone was not enough: a
# ModelingToolkit upgrade left entries that deserialized without error but whose
# registered `avg` no longer resolved, and the breakage only surfaced much later
# as `UndefVarError: avg not defined in ModelingToolkit` inside the ODE solve.
const CACHE_STACK = ("QuantumCumulants", "ModelingToolkit", "Symbolics",
                     "SymbolicUtils", "SciMLBase")

# Versions are read from the loaded modules, not from `Pkg.dependencies()`:
# that walks the whole Manifest and throws when it lists a package the running
# Julia cannot locate (a Manifest written by julia 1.11 lists MbedTLS_jll,
# which 1.12 no longer ships), which silently disabled cache writes on the HPC.
function stack_versions()
    versions = Dict{String, String}()
    for (id, mod) in Base.loaded_modules
        id.name in CACHE_STACK || continue
        v = pkgversion(mod)
        v === nothing || (versions[id.name] = string(v))
    end
    return NamedTuple(Symbol(k) => versions[k] for k in sort(collect(keys(versions))))
end

cache_stamp(order::Integer) = (
    tag = CUMULANT_MODEL_TAG,
    order = order,
    stack = stack_versions(),
    julia = "$(VERSION.major).$(VERSION.minor)",
)

"""Smoke test that a deserialized system still generates working code.

The stamp above should catch every case, but it can only cover packages we
thought to name.  This closes the loop by doing the thing that actually broke:
build an `ODEProblem` from the cached equations and evaluate the right-hand
side once.  It costs a few seconds of code generation against a rebuild that
costs 27 s at order 4 and half an hour at order 7.
"""
function cache_is_usable(eqs)
    try
        @named probe = System(eqs)
        probe = complete(probe)
        u0 = zeros(ComplexF64, length(eqs))
        pmap = CUMULANT_PARAMETERS .=> ComplexF64[10.0, 0.1, 0.04, 0.0, 1e-4, 0.0]
        prob = ODEProblem(probe, merge(Dict(unknowns(probe) .=> u0), Dict(pmap)),
                          (0.0, 1.0))
        du = similar(u0)
        prob.f(du, u0, prob.p, 0.0)
        return all(isfinite, du)
    catch
        return false
    end
end

cache_path(cache_dir::AbstractString, order::Integer) =
    joinpath(cache_dir, "cumulant_eqs_order$(order).jls")

"""
Derive (or load) the scaled, cumulant-truncated equations of motion.

Returns `(eqs_sc, build_time, from_cache)`.  A cache entry is only accepted if
it was written by the same model, QuantumCumulants version, and Julia
minor version; anything else (including a deserialization error, which is the
normal failure mode when a symbolic package is upgraded) triggers a rebuild.
"""
function build_cumulant_system(order::Integer; cache_dir::AbstractString="", verbose::Bool=true)
    path = isempty(cache_dir) ? "" : cache_path(cache_dir, order)

    if !isempty(path) && isfile(path)
        try
            entry = deserialize(path)
            if entry.stamp == cache_stamp(order)
                if cache_is_usable(entry.eqs)
                    verbose && println("order $order: loaded $(length(entry.eqs)) equations from cache")
                    return entry.eqs, 0.0, true
                end
                verbose && println("order $order: cache stamp matches but the " *
                                   "equations no longer generate code, rebuilding")
            else
                verbose && println("order $order: cache stamp mismatch, rebuilding")
            end
        catch err
            verbose && println("order $order: cache unreadable ($(sprint(showerror, err))), rebuilding")
        end
    end

    verbose && println("order $order: deriving (this is the expensive step)...")
    flush(stdout)
    build_time = @elapsed begin
        eqs = meanfield(target_operators(order), H_CUMULANT, J_CUMULANT;
                        rates=RATES_CUMULANT, order=order)
        eqs_c = complete(eqs; filter_func=phase_invariant)
        eqs_sc = scale(eqs_c)
    end
    verbose && @printf("order %d: %d equations derived in %.1f s\n",
                       order, length(eqs_sc), build_time)

    if !isempty(path)
        try
            mkpath(cache_dir)
            serialize(path, (stamp=cache_stamp(order), eqs=eqs_sc))
            verbose && println("order $order: cached to $path")
        catch err
            @warn "could not write cumulant cache (continuing without it)" path exception = err
        end
    end
    return eqs_sc, build_time, false
end
