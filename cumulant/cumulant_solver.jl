using QuantumCumulants
using ModelingToolkit, OrdinaryDiffEq
using LinearAlgebra, Printf
using ForwardDiff

include(joinpath(@__DIR__, "cumulant_model.jl"))

"""
Numerical back end for the QuantumCumulants (indexed, scaled) cumulant
expansion of `cumulant_model.jl`.

The MTK-generated ODE function of QuantumCumulants 0.4 types its parameters
`Complex{Real}` (from `@cnumbers`), so every right-hand-side call is type
unstable and a single steady state costs ~10 s.  Here the MTK right-hand side
is called with a parameter buffer rebuilt as concrete `ComplexF64`, and the
system is wrapped as a real ODE (real and imaginary parts stacked), so the
stiff solvers can use ForwardDiff Jacobians.  Test case (n0=10, P=1e-4,
gA=gD=0, order 4): n = 0.0144464, g2 = 29.9364.

Also provides g2(tau) by quantum regression on the conditional state
rho' = a rho a^dag / <a^dag a>: its moments <X>' = <a^dag X a>/<n> are built
from the steady-state moments with the SAME closure rule the hierarchy uses
(joint cumulants of more than `order` operators set to zero), and then
evolved with the same closed equations, g2(tau) = <a^dag a>'(tau) / <n>.
"""

struct CumulantSystem
    order::Int
    names::Vector{String}
    f!::Function                  # f!(du, u, p) on ComplexF64 vectors
    make_params::Function         # (n0, g, kappa, gA, P, gD) -> p for f!
    n::Int
    idx_n::Int
    idx_n4::Int
    build_time::Float64
end

function load_cumulant_system(order::Integer; cache_dir::AbstractString)
    eqs, build_time, _ = build_cumulant_system(order; cache_dir=cache_dir)
    @named sys = System(eqs)
    sys = complete(sys)
    states = unknowns(sys)
    # MTK's own code generation (it handles the complex symbols and conj()
    # correctly).  The parameter buffer it builds is typed Complex{Real}, which
    # makes every call type unstable; `make_params` rebuilds it with
    # ComplexF64 for each parameter point.
    n = length(states)
    prob = ODEProblem(sys, merge(Dict(states .=> zeros(ComplexF64, n)),
                                 Dict(CUMULANT_PARAMETERS .=> zeros(ComplexF64, 6))),
                      (0.0, 1.0))
    setters = [ModelingToolkit.setp(prob, par) for par in CUMULANT_PARAMETERS]
    template = prob.p
    rhs_mtk = prob.f.f
    f_iip = function (du, u, p)
        rhs_mtk(du, u, p, 0.0)
    end
    make_params = function (vals)
        p = deepcopy(template)
        for (set!, v) in zip(setters, vals)
            set!(p, ComplexF64(v))
        end
        fields = map(concrete_buffer, [getfield(p, k) for k in fieldnames(typeof(p))])
        return typeof(p).name.wrapper(fields...)
    end
    names = [replace(string(s), r"^var\"⟨(.*)⟩\"\(t\)$" => s"\1") for s in states]
    idx_n = findfirst(==("a′*a"), names)
    idx_n4 = findfirst(==("a′*a′*a*a"), names)
    return CumulantSystem(order, names, f_iip, make_params, n, idx_n,
                          something(idx_n4, 0), build_time)
end

concrete_buffer(x::Tuple) = map(concrete_buffer, x)
# NB Complex{Real} counts as a concrete type but is not a bits type; the test
# has to be isbitstype, not isconcretetype.
concrete_buffer(x::AbstractVector{<:Number}) =
    isbitstype(eltype(x)) ? x : ComplexF64.(x)
concrete_buffer(x) = x

# argument order of CUMULANT_PARAMETERS: N g kappa gamma_A P gamma_D
params_vector(cs, n0, g, kappa, gA, P, gD) = cs.make_params((n0, g, kappa, gA, P, gD))

"""Which moments exist for `n0` emitters.

After `scale`, a moment on k distinct emitters feeds the equations of
(k-1)-emitter moments only with the prefactor N-(k-1).  For k > n0 that
prefactor vanishes (k = n0+1) or the moment feeds only other nonexistent
ones, so these moments are decoupled from everything physical -- but their
own equations are not damped and grow without bound (to 1e41 at n0=2),
which wrecks the integrator's step control (n0=1 never finishes).  They are
pinned at zero, which is exact."""
active_moments(cs::CumulantSystem, n0) =
    [count(==('σ'), name) <= n0 for name in cs.names]

function real_rhs(cs::CumulantSystem, p, active=trues(cs.n))
    m = cs.n
    return function (dx, x, _, _)
        u = complex.(view(x, 1:m), view(x, m+1:2m))
        du = similar(u)
        cs.f!(du, u, p)
        @inbounds for i in 1:m
            dx[i] = active[i] ? real(du[i]) : zero(eltype(dx))
            dx[m+i] = active[i] ? imag(du[i]) : zero(eltype(dx))
        end
        nothing
    end
end

to_complex(x, m) = complex.(x[1:m], x[m+1:2m])
to_real(u) = vcat(real.(u), imag.(u))

function residual(cs, p, u, rate_scale, active=trues(cs.n))
    du = similar(u)
    cs.f!(du, u, p)
    return maximum(abs, du[active]) / (rate_scale * max(maximum(abs, u[active]), eps()))
end

"""Largest real part of the Jacobian spectrum at `u` (exact Jacobian of the
real-split system by forward-mode AD; the moments span six decades, so
finite differences are not reliable here).  Negative = stable fixed point,
positive = saddle.

Every moment of the physical solution is either purely real or purely
imaginary (real parameters, phase-invariant initial state), and the
equations preserve that.  The stacked real/imaginary system also contains
the complementary directions -- e.g. an imaginary part of the Hermitian
<a'a> -- which the physical trajectory never enters, and which can carry
spurious growing modes.  The spectrum is therefore taken on the physical
subspace only."""
function max_growth_rate(cs, p, u, active=trues(cs.n))
    f = real_rhs(cs, p, active)
    x = to_real(u)
    J = ForwardDiff.jacobian((dx, y) -> f(dx, y, nothing, 0.0), similar(x), x)
    m = cs.n
    keep = [abs(real(u[i])) >= abs(imag(u[i])) ? i : m + i
            for i in 1:m if active[i]]
    return maximum(real, eigvals(J[keep, keep]))
end

const SS_RESIDUAL_TOL = 1e-8

# ---------------------------------------------------------------------------
# Fixed points by Newton
# ---------------------------------------------------------------------------
#
# Over part of the parameter space the fixed point of the truncated equations
# is linearly unstable (Jacobian eigenvalue with positive real part: near
# threshold, and below threshold for n0 >~ 150).  Integration from the
# physical initial state then approaches it and is driven off again, into a
# finite-time blow-up.  The fixed point itself is perfectly well defined, so
# it is found by Newton's method instead -- seeded from the trajectory, or,
# where the trajectory never gets close, by a homotopy in an auxiliary
# emitter decay.

"""Indices of the physical components in the real-split state: each moment is
purely real (even net photon number) or purely imaginary (odd), and the
equations preserve that.  The complementary components are identically zero
and are left out of the Newton solve (their Jacobian block is singular)."""
function physical_keep(cs::CumulantSystem, active)
    imag_part(name) = (k = parse_moment(name); isodd(abs(k[1] - k[2])))
    [imag_part(cs.names[i]) ? cs.n + i : i for i in 1:cs.n if active[i]]
end

state_residual(cs, p, x, rate_scale, active) =
    residual(cs, p, to_complex(x, cs.n), rate_scale, active)

"""Damped Newton on the physical subspace, in place on the real-split state
`x`.  Returns true once the scaled residual is below `tol`."""
function newton_fixed_point!(cs::CumulantSystem, p, x, active, rate_scale;
                             tol=1e-12, maxit=40)
    f = real_rhs(cs, p, active)
    F! = (dx, y) -> f(dx, y, nothing, 0.0)
    keep = physical_keep(cs, active)
    r = similar(x)
    res = state_residual(cs, p, x, rate_scale, active)
    for _ in 1:maxit
        res < tol && return true
        J = ForwardDiff.jacobian(F!, r, x)
        F!(r, x)
        dx = J[keep, keep] \ r[keep]
        all(isfinite, dx) || return false
        xk = x[keep]; lam = 1.0
        while true                       # backtrack on the scaled residual
            x[keep] .= xk .- lam .* dx
            rn = state_residual(cs, p, x, rate_scale, active)
            if isfinite(rn) && rn < res
                res = rn; break
            end
            lam /= 2
            lam < 1e-6 && (x[keep] .= xk; return false)
        end
    end
    return res < tol
end

function is_physical(cs::CumulantSystem, u)
    ee = real(u[findfirst(==("σee1"), cs.names)])
    return real(u[cs.idx_n]) >= 0 && -1e-12 <= ee <= 1 + 1e-12
end

"""Relative change of <n> and absolute change of <sigma_ee> between two
states; a continuation step that changes them by more than this has jumped
to another branch of the (polynomial) fixed-point equations."""
function jumped(cs::CumulantSystem, u_old, u_new; rel_n=0.2, abs_ee=0.05)
    n_old, n_new = real(u_old[cs.idx_n]), real(u_new[cs.idx_n])
    iee = findfirst(==("σee1"), cs.names)
    return abs(n_new - n_old) > rel_n * max(abs(n_old), 1e-12) ||
           abs(real(u_new[iee] - u_old[iee])) > abs_ee
end

"""Homotopy in an auxiliary emitter decay, gamma_A -> gamma_A + G0 (1-lambda).
At lambda = 0 the emitters are held close to |g>, a product state where the
closure is accurate and the fixed point is stable, so it is reached by
integration; Newton then follows the branch to lambda = 1 with adaptive
steps, rejecting steps that jump branches or leave the physical region.
Returns (x, ok)."""
function homotopy_fixed_point(cs::CumulantSystem, n0, g, kappa, gA, P, gD, active,
                              rate_scale; dlam0=0.05, dlam_min=1e-7)
    G0 = 10 * max(kappa, g, P)
    pars(lam) = params_vector(cs, n0, g, kappa, gA + G0 * (1 - lam), P, gD)
    start = integrate_from_vacuum(cs, pars(0.0), n0, kappa, gA + G0, P, gD, active; g=g)
    start.certified || return start.x, false
    x = start.x
    lam, dlam = 0.0, dlam0
    while lam < 1
        lam_new = min(1.0, lam + dlam)
        xn = copy(x)
        ok = newton_fixed_point!(cs, pars(lam_new), xn, active, rate_scale)
        if ok && is_physical(cs, to_complex(xn, cs.n)) &&
           !jumped(cs, to_complex(x, cs.n), to_complex(xn, cs.n))
            x, lam = xn, lam_new
            dlam *= 1.5
        else
            dlam /= 2
            dlam < dlam_min && return x, false
        end
    end
    return x, true
end

"""Integrate from the vacuum/ground state with a stiff solver, extending the
window until the residual is certified.  Also returns the saved state of
smallest residual along the way (the Newton seed if the run blows up)."""
function integrate_from_vacuum(cs::CumulantSystem, p, n0, kappa, gA, P, gD, active;
                               g=nothing, u0=nothing, alg=Rodas5P())
    rates = filter(>(0), (kappa, gA, P, gD))
    tend = 150.0 / minimum(rates)
    fastest = max(maximum(rates), something(g, 0.0))
    f = real_rhs(cs, p, active)
    x0 = u0 === nothing ? zeros(2cs.n) : to_real(u0)
    best, best_res = copy(x0), Inf
    for ext in 0:4
        prob = ODEProblem(f, x0, (0.0, tend))
        # The answer is accepted on the residual certificate below, not on
        # the integrator tolerance; 1e-8 reproduces the 1e-10 result to
        # ~1e-15 at the test case at a fraction of the cost.  `saveat` only
        # records states for the Newton seed; it does not alter the steps.
        ts = 10 .^ range(log10(tend) - 7, log10(tend), length=200)
        sol = solve(prob, alg; reltol=1e-8, abstol=1e-13, saveat=ts,
                    maxiters=2 * 10^5)
        for xs in sol.u
            r = state_residual(cs, p, xs, fastest, active)
            if isfinite(r) && r < best_res
                best, best_res = copy(xs), r
            end
        end
        sol.retcode == ReturnCode.Success || return (x=best, certified=false, res=best_res)
        x_end = sol.u[end]
        state_residual(cs, p, x_end, fastest, active) <= SS_RESIDUAL_TOL &&
            return (x=copy(x_end), certified=true, res=best_res)
        x0 = x_end; tend *= 5
    end
    return (x=best, certified=false, res=best_res)
end

"""Fixed point of the truncated equations for one parameter point:

  1. integrate from the vacuum/ground state (the physical initial condition)
     and accept a certified stationary end state (route 0);
  2. otherwise Newton from the smallest-residual state of the trajectory,
     accepted if it stays within 5% in <n> of that seed (route 1);
  3. otherwise the auxiliary-decay homotopy (route 2).

Returns a NamedTuple with the complex state, n, g2, residual, status
(0 certified, 2 <a'a'aa> < 0, 4 no fixed point found), route and walltime.
Stability is not part of the status: see `max_growth_rate`."""
function steady_state(cs::CumulantSystem, n0, g, kappa, gA, P, gD;
                      u0=nothing, alg=Rodas5P())
    p = params_vector(cs, n0, g, kappa, gA, P, gD)
    rates = filter(>(0), (kappa, gA, P, gD))
    fastest = max(maximum(rates), g)
    active = active_moments(cs, n0)
    status, route = 0, 0
    local x
    wall = @elapsed begin
        run = integrate_from_vacuum(cs, p, n0, kappa, gA, P, gD, active;
                                    g=g, u0=u0, alg=alg)
        x = run.x
        if !run.certified
            route = 1
            seed_n = x[cs.idx_n]
            ok = newton_fixed_point!(cs, p, x, active, fastest) &&
                 is_physical(cs, to_complex(x, cs.n)) &&
                 abs(x[cs.idx_n] - seed_n) <= 0.05 * abs(seed_n)
            if !ok
                route = 2
                x, ok = homotopy_fixed_point(cs, n0, g, kappa, gA, P, gD, active, fastest)
                ok = ok && newton_fixed_point!(cs, p, x, active, fastest)
            end
            ok || (status = 4)
        end
    end
    u = to_complex(x, cs.n)
    res = status == 4 ? NaN : residual(cs, p, u, fastest, active)
    nph = real(u[cs.idx_n])
    n4 = cs.idx_n4 > 0 ? real(u[cs.idx_n4]) : 2nph^2
    status == 0 && n4 < 0 && (status = 2)
    g2 = status == 4 ? NaN : n4 / nph^2
    return (u=u, n=status == 4 ? NaN : nph, g2=g2, residual=res,
            status=status, route=route, walltime=wall, p=p, active=active)
end

# ---------------------------------------------------------------------------
# Conditional state for g2(tau)
# ---------------------------------------------------------------------------
#
# A moment of the permutation-symmetric, phase-invariant state is fixed by
# (number of a', number of a, multiset of single-atom factors on DISTINCT
# atoms), with factors in {ee, eg, ge}.  All these operators are normally
# ordered and different atoms commute with each other and with the cavity,
# so this key is canonical.

const Key = Tuple{Int,Int,Vector{Symbol}}

function parse_moment(name::AbstractString)::Key
    nd, n = 0, 0; atoms = Symbol[]
    for tok in split(name, '*')
        if tok == "a′"
            nd += 1
        elseif tok == "a"
            n += 1
        else
            mt = match(r"^σ(ee|eg|ge)\d+$", tok)
            mt === nothing && error("cannot parse factor '$tok' in $name")
            push!(atoms, Symbol(mt.captures[1]))
        end
    end
    return (nd, n, sort(atoms))
end

canon(k::Key) = (k[1], k[2], sort(k[3]))
dagger(k::Key) = canon((k[2], k[1],
    [s === :eg ? :ge : s === :ge ? :eg : :ee for s in k[3]]))
phase(k::Key) = k[1] - k[2] + count(==(:eg), k[3]) - count(==(:ge), k[3])
order_of(k::Key) = k[1] + k[2] + length(k[3])

function moment_table(cs::CumulantSystem, u)
    tab = Dict{Key,ComplexF64}()
    for (name, val) in zip(cs.names, u)
        k = parse_moment(name)
        tab[k] = val
        tab[dagger(k)] = conj(val)
    end
    return tab
end

# An operator string is a vector of "factors" (:ad, :a, or (:atom, type, id)).
function key_of(factors)
    nd = count(f -> f === :ad, factors); n = count(f -> f === :a, factors)
    atoms = Symbol[f[2] for f in factors if f isa Tuple]
    return canon((nd, n, atoms))
end

function lookup(tab, factors)
    isempty(factors) && return 1.0 + 0im
    k = key_of(factors)
    phase(k) == 0 || return 0.0 + 0im
    haskey(tab, k) && return tab[k]
    error("moment $(k) of order $(order_of(k)) not available in the hierarchy")
end

# set partitions of 1:n
function partitions(v::Vector{Int})
    length(v) <= 1 && return [[v]]
    first_el, rest = v[1], v[2:end]
    out = Vector{Vector{Vector{Int}}}()
    for p in partitions(rest)
        push!(out, vcat([[first_el]], p))
        for i in eachindex(p)
            q = deepcopy(p); pushfirst!(q[i], first_el); push!(out, q)
        end
    end
    return out
end

function joint_cumulant(tab, factors)
    m = length(factors)
    total = 0.0 + 0im
    for part in partitions(collect(1:m))
        k = length(part)
        prod_ = 1.0 + 0im
        for block in part
            prod_ *= lookup(tab, factors[sort(block)])
            prod_ == 0 && break
        end
        total += (-1)^(k - 1) * factorial(k - 1) * prod_
    end
    return total
end

"""<factors> under the order-M closure: sum over set partitions whose blocks
all have at most M operators of the product of joint cumulants."""
function closed_moment(tab, factors, M)
    length(factors) <= M && return lookup(tab, factors)
    total = 0.0 + 0im
    for part in partitions(collect(1:length(factors)))
        any(b -> length(b) > M, part) && continue
        prod_ = 1.0 + 0im
        for block in part
            prod_ *= joint_cumulant(tab, factors[sort(block)])
            prod_ == 0 && break
        end
        total += prod_
    end
    return total
end

function factors_of(k::Key)
    f = Any[]
    append!(f, fill(:ad, k[1]))
    append!(f, fill(:a, k[2]))
    for (i, s) in enumerate(k[3])
        push!(f, (:atom, s, i))
    end
    return f
end

"""Initial moments of rho' = a rho a^dag / <n>: <X>' = <a^dag X a> / <n>."""
function conditional_state(cs::CumulantSystem, u)
    tab = moment_table(cs, u)
    nph = real(u[cs.idx_n])
    u1 = similar(u)
    for (i, name) in enumerate(cs.names)
        k = parse_moment(name)
        kk = (k[1] + 1, k[2] + 1, k[3])   # a^dag (a'^p a^q) a stays normally ordered
        u1[i] = closed_moment(tab, factors_of(kk), cs.order) / nph
    end
    return u1
end

"""g2(tau) on the grid `taus` (tau[1] must be 0) from steady state `ss`."""
function g2_tau(cs::CumulantSystem, ss, taus; alg=Rodas5P())
    u1 = conditional_state(cs, ss.u)
    u1[.!ss.active] .= 0
    f = real_rhs(cs, ss.p, ss.active)
    prob = ODEProblem(f, to_real(u1), (0.0, taus[end]))
    sol = solve(prob, alg; reltol=1e-10, abstol=1e-14, saveat=taus,
                maxiters=10^6)
    sol.retcode == ReturnCode.Success || @warn "g2(tau) integration" sol.retcode
    return [x[cs.idx_n] for x in sol.u] ./ ss.n, sol.retcode
end
