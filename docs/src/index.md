# ExtensibleMCMC.jl

A modular implementation of the Markov chain Monte Carlo (MCMC) algorithm guided by the principle of ease of its extensibility.

The package comes with enough functionality to:
- run such MCMC algorithms as:
    - ✔/✘ Metropolis-Hastings algorithm (MH) with
        - ✔ Random walk proposals
            - ✔ drawn from `Uniform` on additive and log scale (+ adaptation of steps targeting acceptance rate)
            - ✔ drawn from `Gaussian` on additive and log scale (+ Haario-type adaptation)
        - ✘ Langevin proposals (MALA)
        - ✘ Hamiltionan dynamics
    - ✘ Pseudo-marginal updates
    - ✘ Non-reversible chains
        - ✘ Zig-zag algorithm
- print-out progress information to a console
- save intermediate results to csv files
- provide online diagnostic plots with [ExtensibleMCMCPlots.jl](https://github.com/JuliaDiffusionBayes/ExtensibleMCMCPlots.jl) extension

The package is structured in a way that allows the user to modify the internal flow of the algorithm in a simple way. As a result, the package admits extensions to compute-intensive problems that don't fit neatly into frameworks of other of Julia's MCMC packages.

-------

Depending on your experience and intended use of this package you might consider starting at different places of this documentation.

- For a quick overview of [ExtensibleMCMC.jl](https://juliadiffusionbayes.github.io/ExtensibleMCMC.jl/dev/)'s' main functionality see [Get started](@ref get_started)
- For a systematic introduction to all functionality introduced in this package and information about how to extend it see the [User Manual](@ref manual_start)
- For a didactic introduction to problems that can be solved using [ExtensibleMCMC.jl](https://juliadiffusionbayes.github.io/ExtensibleMCMC.jl/dev/) see the [Tutorials](@ref tutorial_mean_of_bivariate_gsn)
- If you have a problem that you think can be addressed with this package, then check out the [How-to guides](@ref first_how_to) to see if the answer is already there.
