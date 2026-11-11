# Regression tests against numbers reported in the paper.
#   julia --project=. -e 'using Pkg; Pkg.test()'
# Runtime: well under a minute after compilation.

using Test
using LinearAlgebra
using Random
using SuperradiantLMC

@testset "SuperradiantLMC" begin

    @testset "Dicke event table" begin
        # Branching sum rules: local decay conserves the total rate gamma_A n_e,
        # local pump gamma_P (n0 - n_e), for every Dicke state.
        p = LaserParams(7; gamma_P=0.3, gamma_A=0.2, gamma_D=0.1)
        a = zeros(length(DICKE_EVENTS))
        for j2 in 7:-2:1, m2 in -j2:2:j2
            dicke_rates!(a, 3, j2, m2, p)
            ne = (p.n0 + m2) / 2
            @test sum(a[5:7]) ≈ p.gamma_A * ne
            @test sum(a[8:10]) ≈ p.gamma_P * (p.n0 - ne)
            @test all(>=(0), a)
        end
        M, _ = dicke_generator(p, 6)
        @test maximum(abs, sum(M, dims=1)) < 1e-12        # probability conserved
    end

    @testset "Liouvillian is trace preserving" begin
        p = LaserParams(4; gamma_P=0.05, gamma_A=0.01, gamma_D=0.02)
        L, _, trace = symmetry_reduced_liouvillian(p, 8, 4)
        @test norm(transpose(trace) * L, Inf) < 1e-12
    end

    # The collective test case of the paper: n0 = 10, gamma_P = 1e-4,
    # gamma_A = gamma_D = 0, g = 0.1, gamma_c = 0.04 (all ps^-1).
    p = LaserParams(10)

    @testset "test case n0 = 10" begin
        me = solve_exact_me(p)
        @test me.g2 ≈ 20.0381 rtol = 1e-4          # PIQS cross-check: 2e-8
        lmc = solve_dicke_lmc(p)
        @test lmc.g2 ≈ 16.06723228 rtol = 1e-6     # ~20% below the exact ME
        @test lmc.n ≈ me.n rtol = 1e-2             # photon number agrees
        @test lmc.eta_sp < 0.05                    # subradiant storage
        pp = solve_population_polarization(p)
        @test pp.g2 ≈ 20.09742567 rtol = 1e-6
        @test pp.n ≈ 0.01449167941 rtol = 1e-6
        conv = solve_conventional_lmc(p)
        @test conv.g2 ≈ 1.658605186 rtol = 1e-6
        @test conv.n > 1.5 * me.n                  # misses subradiant trapping
    end

    @testset "polarization-order hierarchy" begin
        cutoff = 29
        @test solve_polarization_order(p, cutoff, 1).g2 ≈ 15.89 rtol = 1e-3
        @test solve_polarization_order(p, cutoff, 2).g2 ≈ 20.115 rtol = 1e-3
        @test solve_polarization_order(p, cutoff, 10).g2 ≈ 20.0381 rtol = 1e-4
    end

    @testset "GFRM samples the same chain" begin
        q = LaserParams(3; gamma_P=0.05)
        exact = solve_dicke_lmc(q)
        sampled = gfrm_dicke_lmc(q, 4_000_000; rng=MersenneTwister(1))
        @test sampled.n ≈ exact.n rtol = 0.02
        @test sampled.g2 ≈ exact.g2 rtol = 0.02
    end

    @testset "g2(tau) starts at g2(0)" begin
        q = LaserParams(4)
        segments = [(0.0, 1.0, 0.01)]
        taus, g2 = dicke_lmc_g2_tau(q; segments=segments)
        @test g2[1] ≈ solve_dicke_lmc(q).g2 rtol = 1e-8
        taus, g2 = exact_me_g2_tau(q; segments=segments)
        @test g2[1] ≈ solve_polarization_order(q, 29, 4).g2 rtol = 1e-8
        taus, g2 = population_polarization_g2_tau(q; segments=segments)
        @test g2[1] ≈ solve_population_polarization(q).g2 rtol = 1e-8
    end
end
