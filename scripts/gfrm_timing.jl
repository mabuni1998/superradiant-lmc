# Fig. 7, GFRM series: wall time of GFRM sampling of the Dicke LMC needed to
# reach a 1e-3 relative accuracy in g2(0).
#
#   julia --project=. scripts/gfrm_timing.jl gamma_P=1e-4 [n0=2,3,5,10,20,47,100,200,300,500]
#       [nsteps0=200000] [seeds=3] [maxdouble=14] [target=1e-3]
#
# For every n0 the number of events is doubled from `nsteps0`.  At each step
# count, `seeds` independent trajectories are timed and the RMS relative
# error of their g2(0) is measured against the direct (exact) solution of the
# same chain.  Once the RMS error drops to `target`, the wall time at which
# the target is crossed is interpolated linearly in log(time)-log(error)
# between the last step count above the target and the first one at or below
# it (statistical error ~ 1/sqrt(T) in between).  The reported `walltime` is
# this interpolated time per trajectory; `nsteps` and `rms_rel_err` belong to
# the first step count at or below the target.
#
# The reference g2 is taken from the Dicke-LMC rows of
# data/fig7_timing_P<gamma_P>.tsv when available (the direct solve at
# n0 = 500 takes hours), and computed directly otherwise.
#
# The shipped gfrm_timing_P*.tsv were measured on a laptop with the original
# implementation of the first reaction method (Stochastic.jl); the cost per
# event of `gfrm_dicke_lmc` differs, so rerun timings differ by a constant
# factor, while the flat dependence on n0 is the same.
#
# Writes data/gfrm_timing_P<gamma_P>.tsv.  Runtime: up to ~1 h per n0.

include("common.jl")
using Random

gamma_P = option("gamma_P", P_BELOW)
above = gamma_P > 0.1
n0s = option("n0", above ? [2, 3, 5, 10, 20, 47, 60] : [2, 3, 5, 10, 20, 47, 100, 200, 300, 500])
nsteps0 = option("nsteps0", 200_000)
seeds = option("seeds", 3)
maxdouble = option("maxdouble", 14)
target = option("target", 1e-3)

function reference_g2(n0)
    path = data_path("fig7_timing_P$(gamma_P).tsv")
    if isfile(path)
        _, _, rows = read_tsv(path)
        for r in rows
            r[1] == "lmc" && parse(Int, r[2]) == n0 && return parse(Float64, r[end])
        end
    end
    println("  no stored reference for n0=$n0, solving the chain directly")
    return solve_dicke_lmc(LaserParams(n0; gamma_P=gamma_P)).g2
end

"Mean wall time per trajectory and RMS relative error of g2(0) over `seeds` trajectories."
function measure(p, nsteps, g2ref, k)
    times = Float64[]; errors = Float64[]
    for s in 1:seeds
        rng = MersenneTwister(1000 * (k + 1) + s)
        t = @elapsed r = gfrm_dicke_lmc(p, nsteps; rng=rng)
        push!(times, t); push!(errors, (r.g2 - g2ref) / g2ref)
    end
    return sum(times) / seeds, sqrt(sum(abs2, errors) / seeds)
end

"Time at which the error crosses `target`, log-log interpolation between two measurements."
function crossing_time(t_above, rms_above, t_below, rms_below)
    rms_above == rms_below && return t_below
    frac = (log(target) - log(rms_above)) / (log(rms_below) - log(rms_above))
    return exp(log(t_above) + frac * (log(t_below) - log(t_above)))
end

# Warm-up with the same RNG type as the measurements, so that no compilation
# ends up in the first timed point.
gfrm_dicke_lmc(LaserParams(2; gamma_P=gamma_P), 100_000; rng=MersenneTwister(0))

rows = []
for n0 in n0s
    p = LaserParams(n0; gamma_P=gamma_P)
    g2ref = reference_g2(n0)
    nsteps = nsteps0
    previous = nothing                      # (time, rms) of the last point above target
    for k in 0:maxdouble
        tmean, rms = measure(p, nsteps, g2ref, k)
        @printf("n0=%-5d nsteps=%-11d rms=%.3g  t=%.3g s\n", n0, nsteps, rms, tmean)
        flush(stdout)
        if rms <= target
            if previous === nothing
                println("  target already met at nsteps0; reporting that point (not interpolated)")
                ttarget = tmean
            else
                ttarget = crossing_time(previous..., tmean, rms)
            end
            @printf("  n0=%d: %.4g s to reach %.0e\n", n0, ttarget, target)
            push!(rows, (n0, nsteps, ttarget, g2ref, rms))
            break
        elseif k == maxdouble
            println("  not converged after $maxdouble doublings; reporting the last point")
            push!(rows, (n0, nsteps, tmean, g2ref, rms))
        end
        previous = (tmean, rms)
        nsteps *= 2
    end
end

write_tsv(data_path("gfrm_timing_P$(gamma_P).tsv"),
          "GFRM wall time (s) to a $target RMS relative error in g2(0) against the direct Dicke-LMC solve, " *
          "gamma_P=$gamma_P, gamma_A=gamma_D=0; $seeds seeds, nsteps doubled from $nsteps0, " *
          "crossing interpolated in log-log",
          ["n0", "nsteps", "walltime", "g2_ref", "rms_rel_err"], rows)
