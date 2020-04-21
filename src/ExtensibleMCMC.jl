module ExtensibleMCMC

    using StaticArrays, Distributions, Random, DataStructures, LinearAlgebra
    using Dates

    include("types.jl") # ✔

    const __PREVIOUS = Previous()
    const __PROPOSAL = Proposal()
    const __PRESTEP = PreMCMCStep()
    const __POSTSTEP = PostMCMCStep()

    include("utility_functions.jl") # ✔
    include("priors.jl")
    include("schedule.jl") # ✔
    include("chain_statistics.jl")
    _DIR = "transition_kernels"
    include(joinpath(_DIR, "random_walk.jl")) # ✔/✗ (TODO add a Gaussian random walk and a mixture of two Gaussian random walks, should be simple adaptation of stuff in BridgeSDEInference.jl)
    include(joinpath(_DIR, "adaptation.jl")) # ✔/✗ (TODO add a Haario-type adaptive scheme (and others, see the file for info))

    include("callbacks.jl") # ✔/✗ (TODO coordinate with DiffusionVis.jl to implement a plotting callback)
    include("mcmc.jl") # ✔
    include("run.jl") # ✔
    include("updates.jl") # ✗ (TODO implement conjugate updates, MALA, Hamiltionan Monte Carlo, and much more, this might have do be done after DiffusionMCMC.jl)
    include("workspaces.jl")
    _DIR = "example"
    include(joinpath(_DIR, "gsn_target.jl"))

    export MCMC
    export UniformRandomWalk
    export RandomWalkUpdate
    export GenericMCMCBackend
    export GenericChainStats
    export GsnTargetLaw
    export run!
end # module
