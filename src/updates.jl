#===============================================================================

    Standard parameter updates requiring no knowledge about the underlying
    MCMC backend.
        ✔   Metropolis-Hastings updates with random walk proposals
        ✗   Conjugate updates
        ✗   Metropolis-adjusted Langevin algorithm
        ✗   Hamiltionan Monte Carlo

===============================================================================#
"""
    struct RandomWalkUpdate{TRW,TA,K} <: MCMCParamUpdate
        rw::TRW
        adpt::TA
        loc2glob_idx::K
    end

Definition of an MCMC update that uses Metropolis-Hastings algorithm with random
walk proposals.
"""
struct RandomWalkUpdate{TRW,TA,K,TP} <: MCMCParamUpdate
    rw::TRW
    adpt::TA
    loc2glob_idx::K
    prior::TP

    function RandomWalkUpdate(
            rw::TRW,
            idx_of_global::K,
            prior::TP=ImproperPrior(),
            adpt::TA=NoAdaptation()
        ) where {TRW<:RandomWalk,K,TA,TP}
        new{TRW,TA,K,TP}(rw, adpt, idx_of_global, prior)
    end
end

log_transition_density(updt::RandomWalkUpdate, θ, θ°) = logpdf(updt.rw, θ, θ°)

log_prior(updt::MCMCParamUpdate, θ) = logpdf(updt.prior, θ)

#TODO
struct MALAUpdate <: MCMCGradientBasedUpdate
end

#TODO
struct HamiltonianMCUpdate <: MCMCGradientBasedUpdate
end

#TODO
struct ConjugateGaussianUpdate <: MCMCConjugateUpdate
end


function update!(
        local_ws::LocalWorkspace,
        local_updt::MCMCParamUpdate,
        global_ws::GlobalWorkspace,
        step,
    )
    local_ws.state .= global_ws.state[local_updt.loc2glob_idx]
    local_ws, local_updt
end





#NOTE This is likely to change once conjugate and gradient-based updates are
# implemented and some missing features are discovered
"""
    update!(
        updt::MCMCParamUpdate,
        ws::LocalWorkspace,
        ws_global::GlobalWorkspace,
        step
    )

General recipe for performing parameter update (doing an update of the main
`state` of the MCMC sampler).
"""
function update!(
        updt::MCMCParamUpdate,
        ws::LocalWorkspace,
        global_ws::GlobalWorkspace,
        step,
    )
    proposal!(updt, ws, global_ws, step)
    set_proposal!(global_ws, ws, updt.loc2glob_idx, step) #TODO figure out this
    compute_ll!(global_ws, ws, step)
    accept_reject!(updt, global_ws, ws, step)
    update!(global_ws.stats, global_ws, step)
end

"""
    proposal!(updt::RandomWalkUpdate, ws, global_ws, step)

Sample a proposal value for the MCMC's local `state` variable.
"""
function proposal!(updt::RandomWalkUpdate, ws, global_ws, step)
    rand!(updt.rw, ws.state, ws.state°)
end

# TODO change this
function set_proposal!(
        global_ws::GlobalWorkspace,
        ws::LocalWorkspace,
        loc2glob,
        step
    )
    θ° = global_ws.state_proposal_history[step.mcmciter][step.pidx]
    θ° .= global_ws.state
    θ°[loc2glob] .= ws.state°

    new_parameters!(global_ws.P°, loc2glob, ws.state°)
end

function compute_ll!(
        global_ws::GlobalWorkspace,
        ws::LocalWorkspace,
        step,
    )
    global_ws.ll°[1] = loglikelihood(global_ws.P°, global_ws.data.obs)
    global_ws.ll°_history[step.mcmciter][step.pidx] = global_ws.ll°[1]
end

#NOTE by default accept conjugate updates?
function accept_reject!(updt::MCMCConjugateUpdate, global_ws, ws, step)
    accepted = Val{:accepted}()
    update!(accepted, ws, step)
    update!(accepted, global_ws, step)
end

#NOTE figure out if can be done in full generality
function accept_reject!(updt, global_ws, ws, step)
    ll, ll° = global_ws.ll[1], global_ws.ll°[1]
    llr = (
        ll° - ll
        + log_transition_density(updt, ws.state°, ws.state)
        - log_transition_density(updt, ws.state, ws.state°)
        + log_prior(updt, ws.state°)
        - log_prior(updt, ws.state)
    )
    i = step.mcmciter
    i % 100 == 0 && println("iter: $i, ll: $ll, ll°: $(ll°), llr: $llr")

    accepted = (
        rand(Exponential(1.0)) > -llr ?
        Val(:accepted) :
        Val(:rejected)
    )
    update!(accepted, updt, ws, step)
    update!(accepted, updt, global_ws, step)
end


function update!(::Any, updt, ws::LocalWorkspace, step) end

function update!(::Val{:accepted}, updt, ws::GlobalWorkspace, step)
    ws.acceptance_history[step.mcmciter][step.pidx] = true
    ws.state .= ws.state_proposal_history[step.mcmciter][step.pidx]
    ws.state_history[step.mcmciter][step.pidx] .= ws.state
    ws.ll_history[step.mcmciter][step.pidx] = ws.ll°[1]
    ws.ll[1] = ws.ll°[1]
    new_parameters!(ws.P, ws.state)
end

function update!(::Val{:rejected}, updt, ws::GlobalWorkspace, step)
    ws.acceptance_history[step.mcmciter][step.pidx] = false
    ws.state_history[step.mcmciter][step.pidx] .= ws.state
    ws.ll_history[step.mcmciter][step.pidx] = ws.ll[1]
end
