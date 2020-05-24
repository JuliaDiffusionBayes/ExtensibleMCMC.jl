# Priors
********
At the root of the Bayesian method are prior distributions. In [ExtensibleMCMCPlots.jl](https://github.com/JuliaDiffusionBayes/ExtensibleMCMCPlots.jl) these are represented by structs inheriting from

```@docs
ExtensibleMCMC.Prior
```

Each concrete type must implement a method

```@docs
ExtensibleMCMC.logpdf(::ExtensibleMCMC.Prior, θ)
```

## Concrete types
-------------
The following two priors are important particularly for running quick tests:
```@docs
ExtensibleMCMC.ImproperPrior
ExtensibleMCMC.ImproperPosPrior
```

Additionally, you may leverage a full range of distributions defined in [Distributions.jl](https://juliastats.org/Distributions.jl/stable/) by using

```@docs
ExtensibleMCMC.StandardPrior
ExtensibleMCMC.ProductPrior
```

!!! warning
    These last two are not properly tested, so there might be some hiccups.
