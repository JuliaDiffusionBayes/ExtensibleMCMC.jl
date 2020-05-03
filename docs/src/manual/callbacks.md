# Callbacks
Callbacks are instructions that allow for certain interactions with the state of the sampler while the sampler is still working.

Each concrete implementation of a callback inherits from
```@docs
ExtensibleMCMC.Callback
```

This package implements two types of callbacks:
- `SavingCallback`—for saving intermediate states of the sampler to CSV files
- `REPLCallback`—for printing progress messages to REPL

!!! tip
    See a companion package [ExtensibleMCMCPlots.jl](https://github.com/JuliaDiffusionBayes/ExtensibleMCMCPlots.jl) where we additionally implemented a `PlottingCallback` that does online diagnostic plots that are automatically updated as the chain is being updated.

!!! note
    To pass a list of `Callback`s to an mcmc sampler pass them in a list to a `run!` function.

## Callback for printing progress messages to REPL
```@docs
ExtensibleMCMC.REPLCallback
```

## Callback for saving intermediate results to CSV files
```@docs
ExtensibleMCMC.SavingCallback
```
!!! tip
    If each iteration of your MCMC sampler is very fast then you should avoid appending to CSV files at each iteration of the MCMC chain to prevent slow-downs. Use `save_at_iters` to specify the iterations at which to save to CSV files.

## Writing custom callbacks
Each custom callback may provide interface for the following methods:
```@docs
ExtensibleMCMC.init!(
    ::ExtensibleMCMC.Callback,
    ws::ExtensibleMCMC.GlobalWorkspace
)
ExtensibleMCMC.check_if_execute(
    callback::ExtensibleMCMC.Callback,
    step,
    flag
)
ExtensibleMCMC.execute!(
    callback::ExtensibleMCMC.Callback,
    global_ws::ExtensibleMCMC.GlobalWorkspace,
    local_wss,
    step,
    flag
)
ExtensibleMCMC.cleanup!(
    callback::ExtensibleMCMC.Callback,
    ws::ExtensibleMCMC.GlobalWorkspace,
    step
)
```
