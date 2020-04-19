# ExtensibleMCMC.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://JuliaDiffusionBayes.github.io/ExtensibleMCMC.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://JuliaDiffusionBayes.github.io/ExtensibleMCMC.jl/dev)
[![Build Status](https://travis-ci.com/JuliaDiffusionBayes/ExtensibleMCMC.jl.svg?branch=master)](https://travis-ci.com/JuliaDiffusionBayes/ExtensibleMCMC.jl)

[NOTE this package is currently under development; some functionality explained in the documentation will not work (but it should be indicated where this is the case)]

ExtensibleMCMC.jl
- provides a modular implementation of the **Markov chain Monte Carlo (MCMC)** algorithm
- comes with essential functionality that you'd expect from your MCMC sampler
- is designed to be easily extensible by the users with their own specialized MCMC steps

A case in point is [DiffusionMCMC.jl](https://github.com/JuliaDiffusionBayes/DiffusionMCMC.jl), in which `ExtensibleMCMC.jl` is used to define inference algorithms for diffusion processes.

## Basic functionality
The package comes with enough functionality to run various MCMC algorithms right out of the box, run online diagnostics, print-out progress information to a console, save the data and more. For instance, to define a Metropolis-within-Gibbs algorithm that estimates the mean of a bivariate Gaussian random variable from its observations we should define
- a struct with the `target law` (in our case bivariate Gaussian)
- a function that sets a given subset of the parameters of the target law to new values (entries of a mean vector)
- a log-likelihood function (log-likelihood of bivariate Gaussian)
```julia
using Distributions
using ExtensibleMCMC
const eMCMC = ExtensibleMCMC

mutable struct BivariateGaussian{T}
    P::T
    function BivariateGaussian(μ, Σ)
        @assert length(μ) == 2 && size(Σ) == (2,2)
        P = MvNormal(μ, Σ)
        new{typeof(P)}(P)
    end
end

function eMCMC.new_parameters!(gsn::BivariateGaussian, loc2glob_idx, θ)
    μ, Σ = params(gsn.P)
    μ[loc2glob_idx] .= θ
    gsn.P = MvNormal(μ, Σ)
end

function Distributions.loglikelihood(gsn::BivariateGaussian, observs)
    ll = 0.0
    for obs in observs
        ll += logpdf(gsn.P, obs)
    end
    ll
end
```
That's it, the rest can be handled by standard functions implemented in `ExtensibleMCMC.jl`. We may now generate some data:
```julia
Random.seed!(10)
μ, Σ = [1.0, 2.0], [1.0 0.5; 0.5 1.0]
trgt = MvNormal(μ, Σ)
num_obs = 10

data = (
    P = BivariateGaussian(rand(2), Σ),
    obs = [rand(trgt) for _ in 1:num_obs],
)
```
And then define the parameterization for the Metropolis-within-Gibbs algorithm:
```julia
mcmc_params = (
    mcmc = MCMC(
        [
            RandomWalkUpdate(UniformRandomWalk([1.0]), [1]; prior=ImproperPrior()),
            RandomWalkUpdate(UniformRandomWalk([1.0]), [2]; prior=ImproperPrior()),
        ]
    ),
    num_mcmc_steps = Integer(1e3),
    data = data,
    θinit = [0.0, 0.0],
)
```
In the above `MCMC` allows us to specify a sequence of `updates` that the sampler will cycle through at each iteration of the algorithm. There is a variety (or at least there will be once this package is done) of implemented choices; above, we indicate that we want to have two random walkers, each updating only a single coordinate of the mean vector and going in additive steps according to `+Unif(-1,1)`. The prior on each coordinate is set to be an improper flat prior `π(θᵢ)∝1`. Additionally, we need to set the number of MCMC steps, pass the data (**IMPORTANT** the object with data must contain the `target law` as well as the `observations` in the format of a `NamedTuple` above) and indicate the initial guess for the unknown parameter. We can now run the MCMC sampler:
```julia
workspace = run!(mcmc_params...)
```
The output of the sampler `workspace` is an instance of a `struct` inheriting from `GlobalWorkspace` and contains fields such as `state_history` (the saved MCMC chain), `ll_history` (corresponding evaluations of the log-likelihood function) `acceptance_history` (corresponding indicators for whether the parameter was accepted in the Metropolis-Hastings accept-reject step) and more.

## Other functionality
There are many more standard choices of updates that are implemented in this package. The following are (or are on their way to be) implemented
- [ ] Metropolis-Hastings updates (MH) with
    - [ ] Random walk proposals
        - [x] drawn from `Uniform` on additive and log scale (+ adaptation of steps targeting acceptance rate)
        - [ ] drawn from `Gaussian` on additive and log scale (+ Haario-type adaptation)
    - [ ] Langevin proposals (MALA)
    - [ ] Hamiltionan dynamics
- [ ] Pseudo-marginal updates
- [ ] Non-reversible chains
    - [ ] Zig-zag algorithm

Additionally, the user has access to `Callback`s, which allow for certain interactions with the state of the sampler while it is still working. `SavingCallback` specifies how, where and when should the sampler save intermediate results, `ProgressPrintCallback` specifies the type of messages that should be printed out to REPL informing about the progress of sampling and `PlottingCallback` (that can be called after importing the companion package [ExtensibleMCMCPlots.jl](https://github.com/JuliaDiffusionBayes/ExtensibleMCMCPlots.jl)) makes it possible to produce online diagnostic plots for tracking the progress of MCMC chains visually.

## Purpose of this package
The main purpose of this package is to make it easily extensible, so that **you**, with your fancy specialized algorithm that does some non-standard heavy lifting in one step of an otherwise plain, standard and boring MCMC algorithm can **focus on your thing**, so that once you're done, you can simply plug it into the framework provided by this package and enjoy all the benefits of a swift Julia code combined with a number of generic improvements of MCMC algorithms. The main advantage of extending this package is that it gives you an ultimate control over how you want your `workspace` to look like, and hence, even if your specialized MCMC step requires doing some massive computations, you can still make it as efficient as a custom implementation, because you can design your `Workspace` to operate exclusively on a pre-allocated piece of memory in any way you choose.

## Advanced use
There are 3 main directions in which this package admits extensions:
- Implementing new `Callback`s
- Implementing new `update`s, transition kernels, adaptive schemes, priors
- Implementing new `Workspace`s (and possibly `MCMCSchedule`s)

The first one is about `I/O` improvements and so for many users they can either fall back on the standard implementations and not think about it (or do some light improvements, for instance to print-out some additional information specific to their problem (i.e. their `Workspace`) or plot some imputed random variable etc.) However, any small improvement to standard instance of a `Callback` translates to this improvement being readily enjoyed with **every single call** to MCMC sampler! So don't be too hasty in downplaying the significance of making contributions here.

The second one is about implementing other standard (or non-standard) ingredients of the MCMC sampler that have not been provided in the standard implementation of this package. Maybe you want to implement Riemann manifold MCMC and it's not been done in this package yet---follow the documentation and it should be simple enough.

The third one, the one that motivated existence of this package (and, out of the possibilities above, is perhaps the most invasive path to embark on) relates to providing specialized recipes for performing computations at each step of the MCMC chain. For instance, suppose that you have a specialized algorithm that imputes some data, this allows you to compute gradients with respect to parameters and then you'd like to pass those gradients to a standard, gradient-based update (say MALA) to update the parameters. All of this might be one big complicated mess that you'd be happy to write out programmatically, but you might be reluctant to additionally write an entire MCMC algorithm around it, then a gradient-based update, then care about the efficiency of the entire thing, print-outs, saving functions, online diagnostics functions etc. Only to find out later that, actually, MALA was not the best choice for your example and you wish you could plug-in another gradient-based parameter updating algorithm. This is where `Workspace`s come into play. In essence, these are `structs` with containers that hold any data that is being manipulated and for which you can overload standard `update` or other functions to obtain different than standard behaviour. If you have a large, complicated step that you want to introduce, then define a new `GlobalWorkspace` and/or `LocalWorkspace` and/or overload `MCMCSchedule` and other functions to achieve a desired effect, but leave everything else unchanged so that you can fall back on the standard functionality of the MCMC sampler whenever the sampler is done with executing your specialized piece of code.

To see how to extend the package in any of these three directions, please see the [documentation](https://JuliaDiffusionBayes.github.io/ExtensibleMCMC.jl/dev). For an example of a package that is extends this one see [DiffusionMCMC.jl](https://github.com/JuliaDiffusionBayes/DiffusionMCMC.jl). There, we implement inference for discretely and partially observed diffusion processes.
