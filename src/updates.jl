#===============================================================================

    Standard parameter updates requiring no knowledge about the underlying
    MCMC backend.
        ✔   Metropolis-Hastings updates with random walk proposals
        ✗   Conjugate updates
        ✗   Metropolis-adjusted Langevin algorithm
        ✗   Hamiltionan Monte Carlo

===============================================================================#

#                          HIGHER LEVEL METHODS
#-------------------------------------------------------------------------------

#=
                    these methods MAY be overwritten:
                                                                              =#

"""
    state(ws::GlobalWorkspace, updt::MCMCParamUpdate)

Return a substate of a gloabl state consisting of coordinates that the `updt` is
concerned with.
"""
state(ws::GlobalWorkspace, updt::MCMCParamUpdate) = subidx(state(ws), updt)

state(ws::GlobalWorkspace, updt::MCMCUpdate) = nothing

#                    CUSTOM UPDATES AND LOW LEVEL METHODS
#-------------------------------------------------------------------------------

#=
            These methods MUST be overwritten for each custom
            MCMCParamUpdate
                                                                              =#

"""
    log_transition_density(updt::MCMCParamUpdate, θ, θ°)

Evaluates the log-transition density for an update θ → θ°
"""
function log_transition_density(updt::MCMCParamUpdate, θ, θ°)
    error("log_transition_density not implemented")
end

"""
    proposal!(updt::MCMCParamUpdate, global_ws, ws::LocalWorkspace, step)

Sample a proposal value for the MCMC's local `state` variable.
"""
function proposal!(updt::MCMCParamUpdate, global_ws, ws::LocalWorkspace, step)
    error("proposal! not implemented")
end

"""
    set_parameters!(
        ::Proposal,
        updt::MCMCParamUpdate,
        global_ws::GlobalWorkspace,
        ws::LocalWorkspace,
    )

Set the parameters of the proposal Law to the current value held by the proposal
`state`.
"""
function set_parameters!(
        ::Proposal,
        updt::MCMCParamUpdate,
        global_ws::GlobalWorkspace,
        ws::LocalWorkspace,
    )
    error("set_parameters! for Proposal() is not implemented")
end

"""
    set_parameters!(
        ::Previous,
        updt::MCMCParamUpdate,
        global_ws::GlobalWorkspace,
        ws::LocalWorkspace,
    )

Set the parameters of the currently accepted Law to the current value held by
the accepted `state`.
"""
function set_parameters!(
        ::Previous,
        updt::MCMCParamUpdate,
        global_ws::GlobalWorkspace,
        ws::LocalWorkspace,
    )
    error("set_parameters! for Previous() is not implemented")
end

#=
                    these methods MAY be overwritten:
                                                                              =#

"""
    log_prior(updt::MCMCParamUpdate, θ)

Evaluate log-prior at θ.
"""
log_prior(updt::MCMCParamUpdate, θ) = logpdf(updt.prior, θ)

"""
    coords(updt::MCMCParamUpdate)

Return a list of coordinates that `updt` is concerned with
"""
coords(updt::MCMCParamUpdate) = updt.coords

invcoords(updt::MCMCParamUpdate) = updt.invcoords

"""
    subidx(θ, updt::MCMCParamUpdate)

Return a view of subset of `θ` that `updt` is concerned with
"""
subidx(θ, updt::MCMCParamUpdate) = view(θ, coords(updt))

"""
    compute_gradients_and_momenta!(
        updt::MCMCParamUpdate, ws::LocalWorkspace, ::Any
    )

To be overwritten for methods that are gradient or momentum based.
"""
function compute_gradients_and_momenta!(
        updt::MCMCParamUpdate, ws::LocalWorkspace, ::Any
    )
    nothing
end

#=
                                RandomWalkUpdate
                                                                              =#

"""
    struct RandomWalkUpdate{TRW,TA,K,TP} <: MCMCParamUpdate
        rw::TRW
        adpt::TA
        coords::K
        prior::TP
    end

Definition of an MCMC update that uses Metropolis-Hastings algorithm with random
walk proposals. `rw` is the random walk sampler, `adpt` is the struct
responsible for adaptation of hyperparameters, `coords` lists the
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
    coords::K
    invcoords::Dict{Int64,Int64}
    prior::TP

    function RandomWalkUpdate(
            rw::TRW,
            idx_of_global::K;
            prior::TP=ImproperPrior(),
            adpt::TA=NoAdaptation()
        ) where {TRW<:RandomWalk,K,TA,TP}
        invcoords = Dict{Int64,Int64}()
        for (i,coord) in enumerate(idx_of_global)
            invcoords[coord] = i
        end

        new{TRW,TA,K,TP}(rw, adpt, idx_of_global, invcoords, prior)
    end
end

function readjust!(updt::RandomWalkUpdate, adpt, mcmc_iter)
    readjust!(updt.rw, adpt, mcmc_iter)
end

log_transition_density(updt::RandomWalkUpdate, θ, θ°) = logpdf(updt.rw, θ, θ°)

function proposal!(updt::RandomWalkUpdate, global_ws, ws::LocalWorkspace, step)
    rand!(updt.rw, state(ws), state°(ws))
    while (logpdf(updt.prior, ws.sub_ws°.state) === -Inf)
        rand!(updt.rw, state(ws), state°(ws))
    end
end

function set_parameters!(
        ::Proposal,
        updt::RandomWalkUpdate,
        global_ws::GlobalWorkspace,
        ws::LocalWorkspace,
    )
    set_parameters!(global_ws.P°, coords(updt), state°(ws))
end

function set_parameters!(
        ::Previous,
        updt::RandomWalkUpdate,
        global_ws::GlobalWorkspace,
        ws::LocalWorkspace,
    )
    set_parameters!(global_ws.P, coords(updt), subidx(state(global_ws), updt))
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
