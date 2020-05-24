# [Updates](@id updates_and_decorators)
*********
Updates are structs that hold instructions about the MCMC updates.

## Metropolis–Hastings with random walk proposals
---------------------
The simplest types of updates are the Metropolis-Hastings updates with random walk proposals.
```@docs
ExtensibleMCMC.RandomWalkUpdate
```
A random walk can make steps according to any viable distribution, below we present two cases that have been implemented.
!!! note
    Very often we only need to update a subset of the parameters as in the Metropolis-within-Gibbs algorithm. This is easily done by specifying the indices of relevant coordinates in `idx_of_global` field in the constructor.

### Uniform Random Walker
The simplest type of walker is a *Uniform* random walker.
```@docs
ExtensibleMCMC.UniformRandomWalk
```
!!! note
    In practice, some coordinates of the parameter vector may be restricted to take only positive values. We may impose this restriction by performing a "multiplicative exponentiated uniform random walk" for such coordinates. Such random walk is no longer uniform on the original space, but is uniform in steps for $$\log(X_i)$$.

#### Adaptation
The uniform random walker depends on the hyper-parameter $$ϵ$$, which has a large effect on the learning speed of the MCMC sampler. This hyper-parameter can be learned adaptively in a standard way, by targeting the acceptance rate of the sampler. To turn the adaptation on we have to pass an appropriately initialized `AdaptationUnifRW` as a keyword argument `adpt`:
```@docs
ExtensibleMCMC.AdaptationUnifRW
ExtensibleMCMC.AdaptationUnifRW(θ; kwargs...)
```
To see a simple example illustrating what kind of dramatic difference adaptation can make see the [Tutorial on estimating the mean of a bivariate Gaussian random variable](@ref tutorial_mean_of_bivariate_gsn)

### Gaussian Random Walker
Another implemented type of a random walker is a Gaussian random walker.
```@docs
ExtensibleMCMC.GaussianRandomWalk
```
It can be used for single-site updates as well as joint updates.

#### Adaptation
The adaptation of `GaussianRandomWalk` may no longer be done as in the case of `UniformRandomWalk` (in principle, with the exception of 1-d, but this is not implemented), as the entire covariance matrix needs to be learnt now. This is instead implemented with Haario-type updates.
```@docs
ExtensibleMCMC.HaarioTypeAdaptation
```

In addition, in order to run `HaarioTypeAdaptation`, the `update` transition kernel must be set to `GaussianRandomWalkMix` instead of just `GaussianRandomWalk`, for obvious reasons that follow from the description of the former:
```@docs
ExtensibleMCMC.GaussianRandomWalkMix
```

## (TODO) Metropolis-adjusted Langevin algorithm (MALA)
-----------------

## (TODO) Hamiltonian Monte Carlo
------------

************
***********

# Custom MCMC updates and decorators
-------------
It is easy to define your own update functions. To do so, if your `update` updates parameters (if it does not, then it falls under the category of more advanced updates and the rules might differ) you **must** implement for it
```@docs
ExtensibleMCMC.log_transition_density(updt::ExtensibleMCMC.MCMCParamUpdate, θ, θ°)
ExtensibleMCMC.proposal!(updt::ExtensibleMCMC.MCMCParamUpdate, global_ws, ws::ExtensibleMCMC.LocalWorkspace, step)
ExtensibleMCMC.set_parameters!(
    ::ExtensibleMCMC.Proposal,
    updt::ExtensibleMCMC.MCMCParamUpdate,
    global_ws::ExtensibleMCMC.GlobalWorkspace,
    ws::ExtensibleMCMC.LocalWorkspace,
)
ExtensibleMCMC.set_parameters!(
    ::ExtensibleMCMC.Previous,
    updt::ExtensibleMCMC.MCMCParamUpdate,
    global_ws::ExtensibleMCMC.GlobalWorkspace,
    ws::ExtensibleMCMC.LocalWorkspace,
)
```

#### Optional
Additionally, the following methods are implemented generically, but may be overwritten:
```@docs
ExtensibleMCMC.log_prior(updt::ExtensibleMCMC.MCMCParamUpdate, θ)
ExtensibleMCMC.coords(updt::ExtensibleMCMC.MCMCParamUpdate)
ExtensibleMCMC.subidx(θ, updt::ExtensibleMCMC.MCMCParamUpdate)
ExtensibleMCMC.compute_gradients_and_momenta!(
    updt::ExtensibleMCMC.MCMCParamUpdate, ws::ExtensibleMCMC.LocalWorkspace, ::Any
)
```
