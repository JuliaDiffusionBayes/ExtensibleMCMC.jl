# Overview of `ExtensibleMCMC.jl`'s basic functionality

For simplest problems all that needs to be provided by the user is a `target law`, a function that sets the parameters of the `target law` and a function that evaluates the log-likelihood at the observations. The remaining options can be chosen from a set of objects already pre-defined in this package. Let's look at an example.

## Estimating the mean of a bivariate Guassian

### Defining the target law
In [this folder](https://github.com/JuliaDiffusionBayes/ExtensibleMCMC.jl/tree/master/src/example) we have already defined some examples of the targets laws and a Gaussian law (with a possibility to estimate its mean and covariance matrix) is one of them. For the pedagogical purposes we define the new `BivariateGaussian` struct below.

We start from defining the target law.
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
```
We must provide two functions for it. The first one is a setter of new parameters
```julia
function eMCMC.new_parameters!(gsn::BivariateGaussian, loc2glob_idx, θ)
    μ, Σ = params(gsn.P)
    μ[loc2glob_idx] .= θ
    gsn.P = MvNormal(μ, Σ)
end
```
where `loc2glob_idx` is a vector of indices pointing to parameters that are to be updated. The second one evaluates the log-likelihood:
```julia
function Distributions.loglikelihood(gsn::BivariateGaussian, observs)
    ll = 0.0
    for obs in observs
        ll += logpdf(gsn.P, obs)
    end
    ll
end
```
And we're good to go.

### Generating the data
We may simulate some test data. The important point is that unless you overload the `update` functions yourself, the data should be stored in the format of a `NamedTuple` with fields `P` containing the `target law` and `obs` with a vector of observations.
```julia
using Random
Random.seed!(10)
μ, Σ = [1.0, 2.0], [1.0 0.5; 0.5 1.0]
trgt = MvNormal(μ, Σ)
num_obs = 10

data = (
    P = BivariateGaussian(rand(2), Σ),
    obs = [rand(trgt) for _ in 1:num_obs],
)
```

### Parametrizing the MCMC sampler
Parametrizing the Monte Carlo sampler boils down to specifying a list of `updates` (out of a list provided in this package plus additional updates defined by you) passed as a vector to an `MCMC` struct. Each `update` usually contains
- specification of a `transition kernel`
- an index or indices of a parameter vector that is/are being updated
- a corresponding prior on the parameters
- additional information about adaptation scheme
Additionally, we may pass a `backend` to MCMC to specify what type of `Workspace`s need to be initialized. For now we will not use any adaptation and we will use a default `backend`.

Apart from `MCMC` struct we need to pass the number of mcmc steps, the data and an initial guess for an unknown parameter vector.

Finally, we have an option to pass `Callback`s. For now let's leave these empty. We will discuss them in detail in (...).
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
### Running the sampler
With parameterization of the sampler in place, running it is a one-liner
```julia
workspace = run!(mcmc_params...)
```
