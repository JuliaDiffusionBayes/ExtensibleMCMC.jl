# Updates
Updates are structs that hold instructions about the MCMC updates. Apart from ... each `update` must implement
```@docs
Extensible.log_transition_density(updt::MCMCParamUpdate, θ, θ°)
```

## Metropolis Hastings with random walk proposals
The simplest type of updates are Metropolis-Hastings updates with random walk proposals.
```@docs
ExtensibleMCMC.RandomWalkUpdate
```
The random walk can be any viable distribution, below we present two cases that have been implemented.
!!! note
    Very often we only need to update a subset of the parameters as in the Metropolis-within-Gibbs algorithm. This is easily done by specifying the indices of relevant coordinates in `idx_of_global` field in the constructor.

### Uniform Random Walker
The simplest type of walker is a Uniform random walker. It samples
```math
U_i\sim\texttt{Unif([}-\epsilon,\epsilon\texttt{])},\quad i\in\texttt{idx_of_global},
```
and then, advances each state according to
```math
X_i + U_i,\quad i\in\texttt{idx_of_global}.
```
```@docs
ExtensibleMCMC.UniformRandomWalk
```
!!! note
    In practice, some coordinates of the parameter vector may be restricted to take only positive values. We may impose this restriction by performing a "multiplicative exponentiated uniform random walk" for such coordinates. Such random walk is no longer uniform on the original space, but is uniform in steps for $$\log(X_i)$$.

#### Adaptation
The uniform random walker depends on the hyper-parameter $$ϵ$$, which has a large effect on the learning speed of the MCMC sampler. This hyper-parameter can be learned adaptively in a standard way, by targeting the acceptance rate of the sampler. To turn the adaptation on we have to pass an appropriately initialized `AdaptationUnifRW` as a keyword argument `adpt`:
```@docs
ExtensibleMCMC.AdaptationUnifRW
```
To see a simple example of a difference that adaptation can make see the [Tutorial on estimating the mean of a simple bivariate Gaussian](@ref tutorial_mean_of_bivariate_gsn_adpt)

### Gaussian Random Walker

#### Restricting coordinates to positive values

#### Adaptation

## Metropolis-adjusted Langevin algorithm (MALA)

## Hamiltonian Monte Carlo

# Custom MCMC updates and decorators
