# The idea behind `Workspace`s
Markov chain Monte Carlo algorithms often involve computationally expensive routines. As these often need to be repeated at each MCMC iteration, the MCMC algorithm may be sped up significantly by pre-allocating suitable containers on which all or a majority of the computations are to be performed. In `ExtensibleMCMC.jl` these containers are termed `Workspace`s.

There are two types of `Workspace`s:
1. `GlobalWorkspace`s and
2. `LocalWorkspace`s

## Workspaces inheriting from `GlobalWorkspace`
`GlobalWorkspace`, is the master `Workspace` that is responsible for:
- keeping track of the MCMC chain, in particular the most recent `state` of the chain
- holding the observed data (or at least a pointer to it)
and optionally:
- holding containers for doing computations
- keeping track of some online statistics regarding the chain
- any other object that conceptually belongs to the scope of the chain and not the scope of the local MCMC updates

We implement a generic version of `GlobalWorkspace` that may be suitable for simple problems. For more advanced problems the user will need to implement custom `GlobalWorkspace`s.
```@docs
ExtensibleMCMC.GenericGlobalWorkspace
```

where

```@docs
ExtensibleMCMC.StandardGlobalSubworkspace
```

!!! tip
    To see an example of an implementation of a custom `Workspace` see [DiffusionMCMC.jl](https://github.com/ JuliaDiffusionBayes/DiffusionMCMC.jl), where we implemented `DiffusionGlobalWorkspace` (and `DiffusionLocalWorkspace`).


!!! note
    `StandardGlobalSubworkspace` is simply a collection of the most common fields present in `GlobalWorkspace`. As such, it doesn't need to be present in custom implementations of `GlobalWorkspace`, however, it will often be convenient to do so.


## Workspaces inheriting from `LocalWorkspace`
Each MCMC `update` (for instance `RandomWalkUpdate`) will have its own `LocalWorkspace`. During updates it will have access to both its `LocalWorkspace` as well as the `GlobalWorkspace`, but it will not see `LocalWorkspace`s of other `update`s (however, information between `LocalWorkspace`s may be exchanged prior to each update call). Conceptually, the objects that fall under `LocalWorkspace` are those that
- belong only to a local scope (for instance, proposal `ϑ°` for a subset `ϑ` of all parameters `θ`, or the `∇log-likelihood`)
- provide appropriately shaped views to a global view (for instance, a view to a subset of observations that are to be used for computations in this `update`, or a recipe for how to sub-sample the observations)

Similarly to `GenericGlobalWorkspace`, we implement a generic version of `LocalWorkspace` suitable for simple problems. See [DiffusionMCMC.jl](https://github.com/JuliaDiffusionBayes/DiffusionMCMC.jl) for a more advanced example.

```@docs
ExtensibleMCMC.GenericLocalWorkspace
```
where

```@docs
ExtensibleMCMC.StandardLocalSubworkspace
```


********************************************************************************

# Custom `Workspaces`
One of the most important aspects of `ExtensibleMCMC.jl` is customizability of `Workspace`s. Below, we describe how to define your own `Workspace`s.


## Custom `GlobalWorkspace`
--------------------------------------------------------------------------------
Each `<CUSTOM>GlobalWorskspace` needs to inherit from `GlobalWorkspace`.
```@docs
ExtensibleMCMC.GlobalWorkspace
```
Apart from struct definition we need to provide a `<CUSTOM>Backend` inheriting from:
```@docs
ExtensibleMCMC.MCMCBackend
```
that will help Julia choose suitable initializers.

Additionally, if you plan on re-using some components of `ExtensibleMCMC.jl`, then the following methods **MUST** be defined for your `<CUSTOM>GlobalWorskspace` (using  `<CUSTOM>GlobalWorskspace` in place of `GlobalWorkspace` and `<CUSTOM>Backend` in place of `MCMCBackend`):
```@docs
ExtensibleMCMC.init_global_workspace(
    ::ExtensibleMCMC.MCMCBackend,
    schedule::ExtensibleMCMC.MCMCSchedule,
    updates::Vector{<:ExtensibleMCMC.MCMCUpdate},
    data,
    θinit::Vector{T};
    kwargs...
) where T
ExtensibleMCMC.loglikelihood(ws::ExtensibleMCMC.GlobalWorkspace, ::ExtensibleMCMC.Proposal)
ExtensibleMCMC.loglikelihood(ws::ExtensibleMCMC.GlobalWorkspace, ::ExtensibleMCMC.Previous)

```
!!! note
    When you are overriding `loglikelihood`, then do so via [StatsBase.jl](https://github.com/JuliaDiffusionBayes/StatsBase.jl) or [Distributions.jl](https://github.com/JuliaDiffusionBayes/Distributions.jl) , where the function name originally belongs to, i.e.:

    ```julia
    using StatsBase
    const eMCMC = ExtensibleMCMC
    StatsBase.loglikelihood(ws::CUSTOMGlobalWorkspace, ::eMCMC.Proposal) = ...
    StatsBase.loglikelihood(ws::CUSTOMGlobalWorkspace, ::eMCMC.Previous) = ...
    ```

If `<CUSTOM>GlobalWorskspace` contains a field `sub_ws::StandardGlobalSubworkspace`, then the methods below will work automatically. If it does not, then you should implement them for your `<CUSTOM>GlobalWorskspace`:
```@docs
ExtensibleMCMC.num_mcmc_steps(ws::ExtensibleMCMC.GlobalWorkspace)
ExtensibleMCMC.state(ws::ExtensibleMCMC.GlobalWorkspace)
ExtensibleMCMC.state(ws::ExtensibleMCMC.GlobalWorkspace, step)
ExtensibleMCMC.state°(ws::ExtensibleMCMC.GlobalWorkspace, step)
ExtensibleMCMC.num_updt(ws::ExtensibleMCMC.GlobalWorkspace)
ExtensibleMCMC.estim_mean(ws::ExtensibleMCMC.GlobalWorkspace)
ExtensibleMCMC.estim_cov(ws::ExtensibleMCMC.GlobalWorkspace)
```

# Custom `LocalWorkspace`
--------------------------------------------------------------------------------
Each `<CUSTOM>LocalWorskspace` needs to inherit from `LocalWorkspace`.
```@docs
ExtensibleMCMC.LocalWorkspace
```

Additionally, if you plan on re-using some components of `ExtensibleMCMC.jl`, then the following methods **MUST** be defined for your `<CUSTOM>LocalWorskspace` (using  `<CUSTOM>LocalWorskspace` in place of `LocalWorkspace`):
```@docs
ExtensibleMCMC.create_workspace(
    ::ExtensibleMCMC.MCMCBackend,
    mcmcupdate,
    global_ws::ExtensibleMCMC.GlobalWorkspace,
    num_mcmc_steps
) where {T}
ExtensibleMCMC.accepted(ws::ExtensibleMCMC.LocalWorkspace, i::Int)
ExtensibleMCMC.set_accepted!(ws::ExtensibleMCMC.LocalWorkspace, i::Int, v)
```

If your `<CUSTOM>LocalWorskspace` contains `sub_ws::StandardGlobalSubworkspace` and
`sub_ws°::StandardGlobalSubworkspace` as its fields then the methods below will work automatically:
```@docs
ExtensibleMCMC.ll(ws::ExtensibleMCMC.LocalWorkspace)
ExtensibleMCMC.ll°(ws::ExtensibleMCMC.LocalWorkspace)
ExtensibleMCMC.ll(ws::ExtensibleMCMC.LocalWorkspace, i::Int)
ExtensibleMCMC.ll°(ws::ExtensibleMCMC.LocalWorkspace, i::Int)
ExtensibleMCMC.state(ws::ExtensibleMCMC.LocalWorkspace)
ExtensibleMCMC.state°(ws::ExtensibleMCMC.LocalWorkspace)
```
!!! tip
    The function names associated with proposals end on a character `°`, which in [Atom](https://atom.io/) can be displayed with `\degree` and has a unicode: `U+00B0`.

Finally, there are some functions that are likely to work for custom `LocalWorkspace`s, but you are advised to check for compatibility:
```@docs
ExtensibleMCMC.create_workspaces(v::ExtensibleMCMC.MCMCBackend, mcmc::ExtensibleMCMC.MCMC)
ExtensibleMCMC.llr(ws::ExtensibleMCMC.LocalWorkspace, i::Int)
ExtensibleMCMC.name_of_update(ws::ExtensibleMCMC.LocalWorkspace)
```
