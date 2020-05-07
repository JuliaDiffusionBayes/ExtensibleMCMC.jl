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
        cleanup!(callback, mcmc.workspace, local_wss, (mcmciter=num_mcmc_steps,))
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
        update_callbacks!(callbacks, global_ws, local_wss, step, __PRESTEP)
        update!(local_update, global_ws, local_ws, step)
        update_adaptation!(updates, global_ws, local_ws, step)
        update_callbacks!(callbacks, global_ws, local_wss, step, __POSTSTEP)
    end
end


#            RELATED TO TRANSFER OF INFORMATION BETWEEN WORKSPACES
#-------------------------------------------------------------------------------

"""
    update_workspaces!(
        local_updt::MCMCParamUpdate,
        global_ws::GlobalWorkspace,
        local_ws::LocalWorkspace,
        step,
        prev_ws,
    )
Transfer all the information that is needed at the time of starting the MCMC
update step from the global (or a previously used local) to the currently used
local workspace.
"""
function update_workspaces!(
        local_updt::MCMCParamUpdate,
        global_ws::GlobalWorkspace,
        local_ws::LocalWorkspace,
        step,
        prev_ws,
    )
    state(local_ws) .= state(global_ws, local_updt) #local_ws.sub_ws.state on LHS
    prev_ws == nothing || ( ll(local_ws) .= ll(prev_ws, step.prev_mcmciter) )
    compute_gradients_and_momenta!(local_updt, local_ws, __PREVIOUS)
    local_ws, local_updt
end

#=
function transfer_local_knowledge!(
        local_updt::MCMCParamUpdate,
        local_ws::LocalWorkspace,
        step,
        ::Nothing,
    )
end

function transfer_local_knowledge!(
        local_updt::MCMCParamUpdate,
        local_ws::LocalWorkspace,
        step,
        prev_ws,
    )
    ll(local_ws) .= ll(prev_ws, step.prev_mcmciter)
end
=#

#                          RELATED TO ADAPTATION
#-------------------------------------------------------------------------------

function update_adaptation!(
        updates::AbstractArray{<:MCMCUpdate},
        global_ws::GlobalWorkspace,
        local_ws::LocalWorkspace,
        step
    )
    _accepted = accepted(local_ws, step.mcmciter)

    for (i, updt) in enumerate(updates)
        update_adaptation!(_accepted, updt, global_ws, step, i)
    end
end

function update_adaptation!(
        accepted, updt::MCMCUpdate, global_ws, step, i
    )
    nothing
end

function update_adaptation!(
        accepted::Bool, updt::MCMCParamUpdate, global_ws, step, i
    )
    typeof(updt.adpt) <: NoAdaptation && return
    _my_turn = Val(step.pidx==i)
    register_only_on_my_turn(_my_turn, updt.adpt) && return
    register!(updt, updt.adpt, accepted, state(global_ws, updt))
    ttu = time_to_update(_my_turn, updt.adpt)
    #ttu && println("acceptance rate: ", acceptance_rate(updt.adpt))
    #ttu && println("old ϵ: ", updt.rw.ϵ)
    ttu && readjust!(updt.rw, updt.adpt, step.mcmciter)
    #ttu && println("new ϵ: ", updt.rw.ϵ)
end

# overwrite for your custom adaptations if need be
register_only_on_my_turn(::Val{true}, args...) = false
register_only_on_my_turn(::Val{false}, args...) = true
time_to_update(::Val{false}, args...) = false



#           RELATED TO UPDATE (see the remaining methods in updates.jl)
#-------------------------------------------------------------------------------
#NOTE This is likely to change once conjugate and gradient-based updates are
# implemented and some missing features are discovered
"""
    update!(
        updt::MCMCParamUpdate,
        global_ws::GlobalWorkspace,
        ws::LocalWorkspace,
        step,
    )

General recipe for performing parameter update (doing an update of the main
`state` of the MCMC sampler).
"""
function update!(
        updt::MCMCParamUpdate,
        global_ws::GlobalWorkspace,
        ws::LocalWorkspace,
        step,
    )
    proposal!(updt, global_ws, ws, step) # specific to updt, see `updates.jl`
    set_proposal!(updt, global_ws, ws, step)
    compute_ll!(updt, global_ws, ws, step)
    accept_reject!(updt, global_ws, ws, step)
    update_stats!(global_ws, ws, step)
end

"""
    set_proposal!(
        updt::MCMCParamUpdate,
        global_ws::GlobalWorkspace,
        ws::LocalWorkspace,
        step
    )

The proposal parameter θ° has been sampled by this point. This function
propagates it through workspaces and sets the new parameter in proposal law `P°`
"""
function set_proposal!(
        updt::MCMCParamUpdate,
        global_ws::GlobalWorkspace,
        local_ws::LocalWorkspace,
        step
    )
    θ° = state°(global_ws, step)
    θ° .= state(global_ws)
    subidx(θ°, updt) .= state°(local_ws)

    set_parameters!(__PROPOSAL, updt, global_ws, local_ws)
end

"""
    compute_ll!(
        global_ws::GlobalWorkspace,
        ws::LocalWorkspace,
        step,
    )

Evaluate the proposal log-likelihood at the observations.
"""
function compute_ll!(
        updt::MCMCParamUpdate,
        global_ws::GlobalWorkspace,
        local_ws::LocalWorkspace,
        step,
    )
    ll°(local_ws) .= loglikelihood(global_ws, __PROPOSAL)
    ll°(local_ws, step.mcmciter) .= ll°(local_ws)
    compute_gradients_and_momenta!(updt, local_ws, __PROPOSAL)
end

"""
    accept_reject!(updt, global_ws, ws, step, i=1)

Finish computations of the log-likelihood ratio between the target and proposal
in MH acceptance probability and accept/reject respectively.
"""
function accept_reject!(
        updt, global_ws, ws, step, _ll=sum(ll(ws)), _ll°=sum(ll°(ws)), i=1
    )
    llr = (
        _ll° - _ll
        + log_transition_density(__PROPOSAL, updt, ws, i)
        - log_transition_density(__PREVIOUS, updt, ws, i)
        + log_prior(__PROPOSAL, updt, ws, i)
        - log_prior(__PREVIOUS, updt, ws, i)
    )
    accepted = rand(Exponential(1.0)) > -llr
    register_accept_reject_results!(accepted, updt, global_ws, ws, step, i)
end

#NOTE by default accept conjugate updates?
function accept_reject!(updt::MCMCConjugateUpdate, global_ws, ws, step)
    register_accept_reject_results!(true, updt, global_ws, ws, step, i)
end

"""
    register_accept_reject_results!(
        accepted::Bool,
        updt,
        global_ws::GlobalWorkspace,
        local_ws::LocalWorkspace,
        step,
        i=1
    )
Register the result of accept/reject step.
"""
function register_accept_reject_results!(
        accepted::Bool,
        updt,
        global_ws::GlobalWorkspace,
        local_ws::LocalWorkspace,
        step,
        i=1
    )
    register_accept_reject_results!(accepted, updt, local_ws, step, i)
    set_chain_param!(accepted, updt, global_ws, local_ws, step)
    set_parameters!(__PREVIOUS, updt, global_ws, local_ws)
end

function set_chain_param!(accepted, updt, global_ws, local_ws, step)
    θ = state(global_ws)
    accepted && (
        # update current local and global states
        state(local_ws) .= state°(local_ws);
        subidx(θ, updt) .= state°(local_ws)
    )
    state(global_ws, step) .= θ
end

"""
    register_accept_reject_results!(
        accepted::Bool, updt, ws::LocalWorkspace, step, i=1
    )

Register the results of accept/reject step relevant to a local workspace.
"""
function register_accept_reject_results!(
        accepted::Bool, updt, ws::LocalWorkspace, step, i=1
    )
    ll°(ws, step.mcmciter)[i] = ll°(ws)[i]
    ll(ws, step.mcmciter)[i] = ( accepted ? ll°(ws)[i] : ll(ws)[i] )
    set_accepted!(ws, step.mcmciter, accepted)
end

"""
    log_transition_density(
        ::Previous, updt::MCMCParamUpdate, ws::LocalWorkspace, i
    )

Evaluate the log-density for a transition θ → θ°.
"""
function log_transition_density(
        ::Previous,
        updt::MCMCParamUpdate,
        ws::LocalWorkspace,
        i
    )
    log_transition_density(updt, state(ws), state°(ws))
end

"""
    log_transition_density(
        ::Proposal, updt::MCMCParamUpdate, ws::LocalWorkspace, i
    )

Evaluate the log-density for a transition θ° → θ.
"""
function log_transition_density(
        ::Proposal,
        updt::MCMCParamUpdate,
        ws::LocalWorkspace,
        i
    )
    log_transition_density(updt, state°(ws), state(ws))
end

"""
    log_prior(::Previous, updt::MCMCParamUpdate, ws::LocalWorkspace, i)

Evaluate the log-prior at θ.
"""
function log_prior(::Previous, updt::MCMCParamUpdate, ws::LocalWorkspace, i)
    log_prior(updt, state(ws))
end

"""
    log_prior(::Proposal, updt::MCMCParamUpdate, ws::LocalWorkspace, i)

Evaluate the log-prior at θ°.
"""
function log_prior(::Proposal, updt::MCMCParamUpdate, ws::LocalWorkspace, i)
    log_prior(updt, state°(ws))
end
