#===============================================================================

    Standard parameter updates requiring no knowledge about the underlying
    MCMC backend.
        ✔   Metropolis-Hastings updates with random walk proposals
        ✗   Conjugate updates
        ✗   Metropolis-adjusted Langevin algorithm
        ✗   Hamiltionan Monte Carlo

===============================================================================#
"""
    struct RandomWalkUpdate{TRW,TA,K,TP} <: MCMCParamUpdate
        rw::TRW
        adpt::TA
        loc2glob_idx::K
        prior::TP
    end

Definition of an MCMC update that uses Metropolis-Hastings algorithm with random
walk proposals. `rw` is the random walk sampler, `adpt` is the struct
responsible for adaptation of hyperparameters, `loc2glob_idx` lists the
coordinate indeces of the global `state` that the given update operates on and
`prior` is a prior for this update step (i.e. for a given subset of parameter
vector).

        function RandomWalkUpdate(
            rw::TRW,
            idx_of_global::K;
            prior::TP=ImproperPrior(),
            adpt::TA=NoAdaptation()
        ) where {TRW<:RandomWalk,K,TA,TP}

    Base constructor.
"""
struct RandomWalkUpdate{TRW,TA,K,TP} <: MCMCParamUpdate
    rw::TRW
    adpt::TA
    loc2glob_idx::K
    prior::TP

    function RandomWalkUpdate(
            rw::TRW,
            idx_of_global::K;
            prior::TP=ImproperPrior(),
            adpt::TA=NoAdaptation()
        ) where {TRW<:RandomWalk,K,TA,TP}
        new{TRW,TA,K,TP}(rw, adpt, idx_of_global, prior)
    end
end

"""
    log_transition_density(updt::MCMCParamUpdate, θ, θ°)

Evaluates the log-transition density for an update θ → θ°
"""
function log_transition_density(updt::MCMCParamUpdate, θ, θ°)
    error("log_transition_density not implemented")
end

log_transition_density(updt::RandomWalkUpdate, θ, θ°) = logpdf(updt.rw, θ, θ°)

log_prior(updt::MCMCParamUpdate, θ) = logpdf(updt.prior, θ)

function compute_gradients_and_momenta!(
        updt::MCMCParamUpdate, ws::LocalWorkspace, ::Any
    )
    nothing
end

function update_adaptation!(
        accepted::Bool, updt::MCMCParamUpdate, global_ws, local_ws, step, i
    )
    typeof(updt.adpt) <: NoAdaptation && return
    register!(updt.adpt, accepted, global_ws.sub_ws.state[updt.loc2glob_idx])
    ttu = time_to_update(updt.adpt)
    ttu && println("acceptance rate: ", acceptance_rate(updt.adpt))
    ttu && println("old ϵ: ", updt.rw.ϵ)
    time_to_update(updt.adpt) && readjust!(updt.rw, updt.adpt, step.mcmciter)
    ttu && println("new ϵ: ", updt.rw.ϵ)
end


#TODO
struct MALAUpdate <: MCMCGradientBasedUpdate
end

#TODO
struct HamiltonianMCUpdate <: MCMCGradientBasedUpdate
end

#TODO
struct ConjugateGaussianUpdate <: MCMCConjugateUpdate
end


"""
    update_workspaces!(
        local_updt::MCMCParamUpdate,
        global_ws::GlobalWorkspace,
        local_ws::LocalWorkspace,
        step,
    )
Transfer all the information that is needed at the time of starting the MCMC
update step from the global to the local workspace.
"""
function update_workspaces!(
        local_updt::MCMCParamUpdate,
        global_ws::GlobalWorkspace,
        local_ws::LocalWorkspace,
        step,
        prev_ws,
    )
    local_ws.sub_ws.state .= global_ws.sub_ws.state[local_updt.loc2glob_idx]
    transfer_local_knowledge!(local_updt, local_ws, step, prev_ws)
    compute_gradients_and_momenta!(local_updt, local_ws, __PREVIOUS)
    local_ws, local_updt
end

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
    println()
    local_ws.sub_ws.ll .= prev_ws.sub_ws.ll_history[step.prev_mcmciter]
end


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
    proposal!(updt, global_ws, ws, step)
    set_proposal!(global_ws, ws, updt.loc2glob_idx, step) #TODO figure out this
    compute_ll!(updt, global_ws, ws, step)
    accept_reject!(updt, global_ws, ws, step)
    update_stats!(global_ws.sub_ws.stats, global_ws, ws, step)
end

"""
    proposal!(updt::RandomWalkUpdate, global_ws, ws::LocalWorkspace, step)

Sample a proposal value for the MCMC's local `state` variable.
"""
function proposal!(updt::RandomWalkUpdate, global_ws, ws::LocalWorkspace, step)
    rand!(updt.rw, ws.sub_ws.state, ws.sub_ws°.state)
    while (logpdf(updt.prior, ws.sub_ws°.state) === -Inf)
        rand!(updt.rw, ws.sub_ws.state, ws.sub_ws°.state)
    end
end

# TODO change this
"""
    set_proposal!(
        global_ws::GlobalWorkspace,
        ws::LocalWorkspace,
        loc2glob,
        step
    )

The proposal parameter θ° has been sampled by this point. This function
propagates it through workspaces and sets the new parameter in proposal law `P°`
"""
function set_proposal!(
        global_ws::GlobalWorkspace,
        ws::LocalWorkspace,
        loc2glob,
        step
    )
    θ° = global_ws.sub_ws.state_proposal_history[step.mcmciter][step.pidx]
    θ° .= global_ws.sub_ws.state
    θ°[loc2glob] .= ws.sub_ws°.state

    new_parameters!(global_ws.P°, loc2glob, ws.sub_ws°.state)
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
        ws::LocalWorkspace,
        step,
    )
    ws.sub_ws°.ll[1] = loglikelihood(global_ws.P°, global_ws.sub_ws.data.obs)
    ws.sub_ws°.ll_history[step.mcmciter] .= ws.sub_ws°.ll[1]
    compute_gradients_and_momenta!(updt, ws, __PROPOSAL)
end

#NOTE by default accept conjugate updates?
function accept_reject!(updt::MCMCConjugateUpdate, global_ws, ws, step)
    register_accept_reject_results!(true, updt, global_ws, ws, step, i)
end

"""
    accept_reject!(updt, global_ws, ws, step, i=1)

Finish computations of the log-likelihood ratio between the target and proposal
in MH acceptance probability and accept/reject respectively.
"""
function accept_reject!(
        updt, global_ws, ws, step, ll=ws.sub_ws.ll[1], ll°=ws.sub_ws°.ll[1], i=1
    )
    llr = (
        ll° - ll
        + log_transition_density(__PROPOSAL, updt, ws)
        - log_transition_density(__PREVIOUS, updt, ws)
        + log_prior(__PROPOSAL, updt, ws)
        - log_prior(__PREVIOUS, updt, ws)
    )
    # move printouts to callbacks
    j = step.mcmciter
    j % 100 == 0 && println("iter: $j, ll: $ll, ll°: $(ll°), llr: $llr")

    accepted = rand(Exponential(1.0)) > -llr
    register_accept_reject_results!(accepted, updt, global_ws, ws, step, i)
end

"""
    log_transition_density(::Previous, updt::MCMCParamUpdate, ws::LocalWorkspace)

Evaluate the log-density for a transition θ → θ°.
"""
function log_transition_density(
        ::Previous,
        updt::MCMCParamUpdate,
        ws::LocalWorkspace
    )
    log_transition_density(updt, ws.sub_ws.state, ws.sub_ws°.state)
end

"""
    log_transition_density(
        ::Proposal,
        updt::MCMCParamUpdate,
        ws::LocalWorkspace
    )

Evaluate the log-density for a transition θ° → θ.
"""
function log_transition_density(
        ::Proposal,
        updt::MCMCParamUpdate,
        ws::LocalWorkspace,
    )
    log_transition_density(updt, ws.sub_ws°.state, ws.sub_ws.state)
end

"""
    log_prior(updt::MCMCParamUpdate, ws::LocalWorkspace)

Evaluate the log-prior at θ.
"""
function log_prior(::Previous, updt::MCMCParamUpdate, ws::LocalWorkspace)
    log_prior(updt, ws.sub_ws.state)
end

"""
    log_prior(updt::MCMCParamUpdate, ws::LocalWorkspace, ::Proposal)

Evaluate the log-prior at θ°.
"""
function log_prior(::Proposal, updt::MCMCParamUpdate, ws::LocalWorkspace)
    log_prior(updt, ws.sub_ws°.state)
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
    θ = global_ws.sub_ws.state
    accepted && ( θ[updt.loc2glob_idx] .= local_ws.sub_ws°.state )
    global_ws.sub_ws.state_history[step.mcmciter][step.pidx] .= θ
    new_parameters!(global_ws.P, θ)
    update_adaptation!(accepted, updt, global_ws, local_ws, step, i)
end

new_parameters!(P, θ) = new_parameters!(P, 1:length(θ), θ)

"""
    register_accept_reject_results!(
        accepted::Bool, updt, ws::LocalWorkspace, step, i=1
    )

Register the results of accept/reject step relevant to a local workspace.
"""
function register_accept_reject_results!(
        accepted::Bool, updt, ws::LocalWorkspace, step, i=1
    )
    ws.sub_ws°.ll_history[step.mcmciter][i] = ws.sub_ws°.ll[i]
    ws.sub_ws.ll_history[step.mcmciter][i] = (
        accepted ? ws.sub_ws°.ll[i] : ws.sub_ws.ll[i]
    )
    ws.acceptance_history[step.mcmciter] = accepted
end
