module ExtensibleMCMC

    using StaticArrays, Distributions, Random, DataStructures, LinearAlgebra
    using Dates

    import Distributions: logpdf

    include("types.jl") # ✔

    const __PREVIOUS = Previous()
    const __PROPOSAL = Proposal()
    const __PRESTEP = PreMCMCStep()
    const __POSTSTEP = PostMCMCStep()

    include("utility_functions.jl") # ✔
    include("priors.jl")
    include("schedule.jl") # ✔
    include("chain_statistics.jl") # ✔
    _DIR = "transition_kernels"
    include(joinpath(_DIR, "random_walk.jl")) # ✔
    include(joinpath(_DIR, "adaptation.jl")) # ✔

    include("callbacks.jl") # ✔
    include("mcmc.jl") # ✔
    include("run.jl") # ✔
    include("updates.jl") # ✔/✗ (TODO implement conjugate updates, MALA, Hamiltionan Monte Carlo, and much more, this might have do be done after DiffusionMCMC.jl)
    include("workspaces.jl") # ✔
    _DIR = "example"
    include(joinpath(_DIR, "gsn_target.jl")) # ✔

    export MCMC
    export UniformRandomWalk, GaussianRandomWalk, GaussianRandomWalkMix
    export AdaptationUnifRW, HaarioTypeAdaptation
    export RandomWalkUpdate
    export GenericMCMCBackend
    export GenericChainStats
    export GsnTargetLaw
    export run!
    export get_decorators, isdecorator
    export ImproperPosPrior, ImproperPrior
    export SavingCallback, REPLCallback
end # module
