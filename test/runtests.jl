using ExtensibleMCMC
using Test, StaticArrays, Distributions
const eMCMC = ExtensibleMCMC

@testset "schedule" begin
    schedule_params = (
        num_mcmc_iter = 10,
        num_params = 4,
        exclude_params = [(1,3:8), (2,4:2:10)],
    )
    schedule = eMCMC.MCMCSchedule(schedule_params...)

    expected = (
        (1,1),  (1,2),  (1,3),  (1,4),
        (2,1),  (2,2),  (2,3),  (2,4),
                (3,2),  (3,3),  (3,4),
                        (4,3),  (4,4),
                (5,2),  (5,3),  (5,4),  (5,5),  (5,6),
                        (6,3),          (6,5),  (6,6),
                (7,2),  (7,3),          (7,5),  (7,6),
                        (8,3),                  (8,6),
        (9,1),  (9,2),  (9,3),                  (9,6),
        (10,1),         (10,3),         (10,5), (10,6),
    )

    for (i,s) in enumerate(schedule)
        @test expected[i] == Tuple(s)
        s.mcmciter == 5 && s.pidx == 3 && eMCMC.reschedule!(
            schedule, 2, [4], [(5, 8:9)]
        )
    end
end

@testset "adaptation for random walk" begin
    template = eMCMC.AdaptationUnifRW{Float64}(0.234,100,1.0,1.0e-12,1e7,1e2,1)
    @test template == eMCMC.AdaptationUnifRW(1.0)
    @test template == eMCMC.AdaptationUnifRW([2.0])
    @test template == eMCMC.AdaptationUnifRW(SVector{1}(3.0))

    longerθ = eMCMC.AdaptationUnifRW([1.0, 2.0])
    @test template != longerθ
    @test eMCMC.isequal_except(template, longerθ, :N)

    longer_staticθ = eMCMC.AdaptationUnifRW(SVector{3}(1.0, 2.0, 3.0))
    @test template != longer_staticθ
    @test eMCMC.isequal_except(template, longer_staticθ, :N)

    new_scale = eMCMC.AdaptationUnifRW(1.0; scale=3.0)
    @test template != new_scale
    @test eMCMC.isequal_except(template, new_scale, :scale)

    new_params = eMCMC.AdaptationUnifRW(1.0; scale=3.0, target_accpt_rate=0.111,min=10.0)
    @test template != new_scale
    @test eMCMC.isequal_except(template, new_scale, :scale, :target_accpt_rate, :min)
    @test new_params.scale == 3.0
    @test new_params.target_accpt_rate == 0.111
    @test new_params.min == 10.0

    ar_vec = eMCMC.AdaptationUnifRW([1.0, 2.0]; scale=[3.0, 4.0], target_accpt_rate=0.111, min=10.0)
    @test ar_vec.target_accpt_rate == 0.111
    @test ar_vec.min == [10.0, 10.0]
    @test ar_vec.max == [1e7, 1e7]
    @test ar_vec.scale == [3.0, 4.0]
    @test ar_vec.offset == [100.0, 100.0]
    @test ar_vec.N == 2
    @test ar_vec.adapt_every_k_steps == 100

    ar_vec2 = eMCMC.AdaptationUnifRW([1.0, 2.0]; scale=[3.0, 4.0], target_accpt_rate=0.111, min=[10.0, 10.0])
    @test ar_vec == ar_vec2

    ar_svec = eMCMC.AdaptationUnifRW(SVector{2}([1.0, 2.0]); scale=[3.0, 4.0], target_accpt_rate=0.111, min=10.0)
    @test ar_svec.target_accpt_rate == 0.111
    @test ar_svec.min == @SVector [10.0, 10.0]
    @test ar_svec.max == @SVector [1e7, 1e7]
    @test ar_svec.scale == @SVector [3.0, 4.0]
    @test ar_svec.offset == @SVector [100.0, 100.0]
    @test ar_svec.N == 2
    @test ar_svec.adapt_every_k_steps == 100

    ar_svec2 = eMCMC.AdaptationUnifRW(SVector{2}([1.0, 2.0]); scale=[3.0, 4.0], target_accpt_rate=0.111, min=SVector{2}([10.0, 10.0]))
    @test ar_svec == ar_svec2

    ar_svec3 = eMCMC.AdaptationUnifRW(SVector{2}([1.0, 2.0]); scale=[3.0, 4.0], target_accpt_rate=0.111, min=[10.0, 10.0])
    @test ar_svec2 == ar_svec3
end

@testset "mcmc" begin
    # estiate mean of a normal distribution
    μ = [1.0, 2.0]
    Σ = [1.0 0.5; 0.5 1.0]

    #Random.seed!(10)
    trgt = MvNormal(μ, Σ)
    num_obs = 10

    mcmc_params = (
        mcmc = MCMC(
            [
                RandomWalkUpdate(UniformRandomWalk([1.0]), [1]),
                RandomWalkUpdate(UniformRandomWalk([1.0]), [2]),
            ];
            backend=GenericMCMCBackend(), # this is a default anyway
        ),
        num_mcmc_steps = Integer(1e3),
        data = (
            P = GsnTargetLaw(μ, Σ),
            obs = [rand(trgt) for _ in 1:num_obs],
        ),
        θinit = [0.0, 0.0],
        callbacks = eMCMC.Callback[],
    )

    ws = run!(mcmc_params...)
end
