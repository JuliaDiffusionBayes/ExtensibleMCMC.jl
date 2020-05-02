#===============================================================================

        Defines the main workhorse functions of this package that perform
        MCMC sampling for stochastic processes observed discretely in time.

===============================================================================#

"""
    run!(mcmc::MCMC, num_mcmc_steps, data, θinit, callbacks; kwargs...)

The main calling function of this package that initializes, runs and outputs the
results of an MCMC sampler for discretely observed stochastic processes. `mcmc`
specifies the sequence of updates that constitute a single MCMC step and
provides additional info about the MCMC backend algorithm to be used.
`num_mcmc_steps` is the total number of MCMC steps, `data` completely
characterizes everything that is known about the observations (including
everything known about the underlying dynamics and the way it was collected),
`θinit` is the initial guess for the unknown parameters and `callbacks` is a
collection of extra utility functions that do extra work around MCMC sampling.
`exclude_updates` is a standard named argument which lists the update indices
and corresponding ranges of MCMC iterations at which given updates are supposed
to be omitted. Additional named arguments are passed onto initializers of global
workspace and depend on the chosen `MCMCBackend`.
"""
function run!(mcmc::MCMC, num_mcmc_steps, data, θinit, callbacks=Callback[]; kwargs...)
    init!(
        mcmc,
        num_mcmc_steps,
        data,
        θinit,
        get(kwargs, :exclude_updates, []);
        kwargs...
    )
    local_wss = create_workspaces(mcmc.backend, mcmc)

    init!(callbacks, mcmc.workspace)
    __run!(mcmc.workspace, local_wss, mcmc.updates, mcmc.schedule, callbacks)
    for callback in callbacks
        cleanup!(callback, mcmc.workspace, num_mcmc_steps)
    end
    mcmc.workspace, local_wss
end

"""
    __run!(global_ws, local_wss, updates, schedule, callbacks)

Internal loops that run the (already initialized) MCMC sampler. `global_ws` and
`local_wss` are the already initialized global workspace and local workspaces
(one for each update) respectively. `schedule` is the iterator over MCMC steps
and `callbacks` is a list of callbacks.
"""
function __run!(global_ws, local_wss, updates, schedule, callbacks)
    #TODO find out whether try-catch is the best way to register user's
    # interruptions for initiating an early exit from the loop.
    #TODO implement functionality that registers user interruption,
    # prompts for confirmation of early exit and whether to save the most recent
    # state.
    for step in schedule
        local_ws, local_update = update_workspaces!(
            updates[step.pidx],
            global_ws,
            local_wss[step.pidx],
            step,
            (step.prev_pidx===nothing ? nothing : local_wss[step.prev_pidx])
        )
        update!(callbacks, global_ws, local_ws, step, __PRESTEP)
        update!(local_update, global_ws, local_ws, step)
        update_adaptation!(updates, global_ws, local_ws, step)
        update!(callbacks, global_ws, local_ws, step, __POSTSTEP)
    end
end
