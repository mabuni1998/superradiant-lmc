# Linear-algebra kernels shared by all direct solvers:
#   * stationary state of a sparse generator by shifted inverse iteration
#   * Crank--Nicolson propagation for delayed correlations g2(tau)

"""
    stationary_state(M, trace, start; atol, maxiter=200, shift=nothing)

Stationary vector `x` of the sparse generator `M` (dx/dt = M x), normalized
so that `dot(trace, x) == 1`.

All eigenvalues of a trace-preserving generator have non-positive real part,
so `shift*I - M` is nonsingular for any `shift > 0`.  It is factorized once,
and repeated application of its inverse projects onto the null vector; each
iteration damps the non-stationary components by `shift/(shift + gap)`.  In
contrast to replacing one row of `M` by the normalization condition, this
keeps the matrix sparse, so the LU factorization scales to millions of
states.  The default shift is `1e-10` times the largest diagonal rate.
"""
function stationary_state(M::SparseMatrixCSC, trace::AbstractVector, start::AbstractVector;
                          atol, maxiter=200, shift=nothing)
    size(M, 1) == size(M, 2) || throw(DimensionMismatch("generator must be square"))
    rate_scale = maximum(abs, diag(M))
    trace_error = norm(transpose(trace) * M, Inf)
    trace_error <= 1e-10 * max(1.0, rate_scale) ||
        error("generator is not trace preserving (error $trace_error)")

    sigma = isnothing(shift) ? 1e-10 * rate_scale : float(shift)
    F = lu(sigma * I - M)
    x = start ./ dot(trace, start)
    residual = Inf
    for _ in 1:maxiter
        x = F \ x
        x ./= dot(trace, x)
        residual = norm(M * x, Inf)
        residual <= atol && break
    end
    residual <= atol || error("stationary residual $residual exceeds $atol")
    return x, residual
end

"""
Piecewise-uniform time grid for g2(tau) (start, stop, step) in ps: fine steps
at short delays (collective Rabi oscillations), coarse steps at long delays
(pump recovery ~ 1/gamma_P).
"""
const G2TAU_SEGMENTS = [
    (0.0, 2.0, 0.005),
    (2.0, 50.0, 0.05),
    (50.0, 500.0, 0.5),
    (500.0, 5000.0, 5.0),
    (5000.0, 50000.0, 25.0),
]

"""
    propagate(M, x0, observable; segments=G2TAU_SEGMENTS, outputs_per_segment=60)

Integrate dx/dt = M x from `x0` with the Crank--Nicolson scheme and record
`observable(x)` about `outputs_per_segment` times per segment.  Each segment
reuses one sparse LU factorization of `I - dt/2 M`.  Returns `(taus, values)`.
"""
function propagate(M, x0, observable; segments=G2TAU_SEGMENTS, outputs_per_segment=60)
    taus = Float64[0.0]
    values = [observable(x0)]
    x = copy(x0)
    for (t0, t1, dt) in segments
        nsteps = round(Int, (t1 - t0) / dt)
        Fplus = I + (dt / 2) * M
        Fminus = lu(I - (dt / 2) * M)
        stride = max(1, nsteps ÷ outputs_per_segment)
        for k in 1:nsteps
            x = Fminus \ (Fplus * x)
            if k % stride == 0 || k == nsteps
                push!(taus, t0 + k * dt)
                push!(values, observable(x))
            end
        end
    end
    return taus, values
end
