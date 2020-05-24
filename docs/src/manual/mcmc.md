# MCMC
******

The main object with which the user declares the type of MCMC algorithm to use is
```@docs
ExtensibleMCMC.MCMC
```
The constructor of `MCMC` accepts a list of `update`s that *de facto* define the MCMC algorithm. To learn more about them see the section on [update functions](@ref updates_and_decorators).

The main algorithm may by run by passing an instance of `MCMC`, together with some additional parameters to the function
```@docs
ExtensibleMCMC.run!
```
The output of `run!` is a tuple of a global workspace and local workspaces. These can be used to query such objects as a chain of accepted parameters, a chain of proposed parameters, corresponding log-likelihood functions and more. See the section on [Workspaces](@ref workspaces_explained) to learn more.
