#===============================================================================

                        Definitions of Workspaces
        Defines *structs*:
        - StandardGlobalSubworkspace - convenience container that is most likely
                             going to be used even with custom global Workspaces
        - StandardLocalSubworkspace - convenience container that is most likely
                             going to be used even with custom local Workspaces
        - GenericGlobalWorkspace - simple global Workspace for simple problems
        - GenericLocalWorkspace - simple local Workspace for simple problems
        As well as functions:
        - init_global_workspace
        - create_workspaces

===============================================================================#

#                              GLOBAL WORKSPACE
#-------------------------------------------------------------------------------

#=        |-------------------------------------------------------|
          |    each custom GlobalWorkspace MUST overwrite these:  |
          |-------------------------------------------------------|           =#
"""
    init_global_workspace(
        ::MCMCBackend,
        num_mcmc_steps,
        updates::Vector{<:MCMCUpdate},
        data,
        θinit::Vector{T};
        kwargs...
    ) where T

Initialize the `<custom>GlobalWorkspace`. `<custom>MCMCBackend` points to which
`GlobalWorkspace` constructors to use, `updates` is a list of MCMC updates,
`θinit` is the initial guess for the parameter and `kwargs` are the named
arguments passed to `run!`.
"""
function init_global_workspace(
        ::MCMCBackend,
        num_mcmc_steps,
        updates::Vector{<:MCMCUpdate},
        data,
        θinit::Vector{T};
        kwargs...
    ) where T
    error("init_global_workspace not implemented")
end

"""
    loglikelihood(ws::GlobalWorkspace, ::Proposal)

Evaluate the loglikelihood for the proposal Law and observations stored in a
global workspace.
"""
function Distributions.loglikelihood(ws::GlobalWorkspace, ::Proposal)
    error(
        string(
            "loglikelihood(ws::GlobalWorkspace, ::Proposal) for a custom ",
            "GlobalWorkspace is not implemented"
        )
    )
end

"""
    loglikelihood(ws::GlobalWorkspace, ::Previous)

Evaluate the loglikelihood for the accepted Law and observations stored in a
global workspace.
"""
function Distributions.loglikelihood(ws::GlobalWorkspace, ::Previous)
    error(
        string(
            "loglikelihood(ws::GlobalWorkspace, ::Previous) for a custom ",
            "GlobalWorkspace is not implemented"
        )
    )
end


#=
        for custom GlobalWorkspaces these methods will work so long as
        they contain sub_ws::StandardGlobalSubworkspace, otherwise they
        need to be overwritten.
                                                                              =#

"""
    num_mcmc_steps(ws::GlobalWorkspace)

Return the total set number of MCMC iterations.
"""
num_mcmc_steps(ws::GlobalWorkspace) = num_mcmc_steps(ws.sub_ws)

"""
    state(ws::GlobalWorkspace)

Return currently accepted state of the chain.
"""
state(ws::GlobalWorkspace) = state(ws.sub_ws)

"""
    state(ws::GlobalWorkspace, step)

Return the state of the chain accepted at the the `step` iteration of the Markov
chain.
"""
state(ws::GlobalWorkspace, step) = state(ws.sub_ws, step)

"""
    state°(ws::GlobalWorkspace, step)

Return the state of the chain proposed at the the `step` iteration of the Markov
chain.
"""
state°(ws::GlobalWorkspace, step) = state°(ws.sub_ws, step)

"""
    num_updt(ws::GlobalWorkspace)

Return the total set number of MCMC updates that may be performed at each MCMC
iteration.
"""
num_updt(ws::GlobalWorkspace) = num_updt(ws.sub_ws)

"""
    estim_mean(ws::GlobalWorkspace)

Return the empirical mean of the parameter.
"""
estim_mean(ws::GlobalWorkspace) = estim_mean(ws.sub_ws)

"""
    estim_cov(ws::GlobalWorkspace)

Return the empirical covariance of the parameter.
"""
estim_cov(ws::GlobalWorkspace) = estim_cov(ws.sub_ws)

#=
                            standard sub-Workspace
                                                                              =#

"""
    struct StandardGlobalSubworkspace{T,TD} <: GlobalWorkspace{T}
        state::Vector{T}
        state_history::Vector{Vector{Vector{T}}}
        state_proposal_history::Vector{Vector{Vector{T}}}
        data::TD
        stats::GenericChainStats{T}
    end

Standard containers expected to be present in every global workspace. `state` is
the currently accepted parameter θ, `state_history` is a chain of `state`s that
have been accepted and `state_proposal_history` is a chain of `state`s that have
been proposed. `data` are the data passed to an MCMC sampler (usually just a
pointer) and `stats` gathers some basic online information about the chain.
"""
struct StandardGlobalSubworkspace{T,TD} <: GlobalWorkspace{T}
    state::Vector{T}
    state_history::Vector{Vector{Vector{T}}}
    state_proposal_history::Vector{Vector{Vector{T}}}
    data::TD
    stats::GenericChainStats{T}

    function StandardGlobalSubworkspace(
            num_mcmc_steps,
            num_updates,
            data::TD,
            θinit::Vector{T},
        ) where {T,TD}
        M, NU = num_mcmc_steps, num_updates # name alias
        ∅s = [Vector{T}(undef, length(θinit)) for _=1:NU]
        new{T,TD}(
            deepcopy(θinit),
            [deepcopy(∅s) for _=1:M],
            [deepcopy(∅s) for _=1:M],
            data,
            GenericChainStats(θinit, NU, M),
        )
    end
end

num_mcmc_steps(ws::StandardGlobalSubworkspace) = length(ws.state_history)
state(ws::StandardGlobalSubworkspace) = ws.state
function state(ws::StandardGlobalSubworkspace, step)
    ws.state_history[step.mcmciter][step.pidx]
end
function state°(ws::StandardGlobalSubworkspace, step)
    ws.state_proposal_history[step.mcmciter][step.pidx]
end
num_updt(ws::StandardGlobalSubworkspace) = length(first(ws.state_history))
estim_mean(ws::StandardGlobalSubworkspace) = ws.stats.mean
estim_cov(ws::StandardGlobalSubworkspace) = ws.stats.cov

#=
                        simple, custom GlobalWorkspace
                                                                              =#

"""
    struct GenericGlobalWorkspace{T,TD,TL} <: GlobalWorkspace{T}
        sub_ws::StandardGlobalSubworkspace{T,TD}
        P::TL
        P°::TL
    end

Generic global workspace with `sub_ws` containing current `state` and keeping
track of its history and some basic statistics. `P` and `P°` are the target laws
with accepted and proposal `state` set as parameters.
"""
struct GenericGlobalWorkspace{T,TD,TL} <: GlobalWorkspace{T}
    sub_ws::StandardGlobalSubworkspace{T,TD}
    P::TL
    P°::TL
end

function init_global_workspace(
        ::GenericMCMCBackend,
        num_mcmc_steps,
        updates::Vector{<:MCMCUpdate},
        data,
        θinit::Vector{T};
        kwargs...
    ) where T
    sub_ws = StandardGlobalSubworkspace(
        num_mcmc_steps,
        length(updates),
        data,
        θinit,
    )
    GenericGlobalWorkspace{T,typeof(data),typeof(data.P)}(
        sub_ws,
        deepcopy(data.P),
        deepcopy(data.P),
    )
end

function Distributions.loglikelihood(ws::GenericGlobalWorkspace, ::Proposal)
    loglikelihood(ws.P°, ws.sub_ws.data.obs)
end

#=
                generic methods for all GlobalWorkspaces
                                                                              =#

function Base.summary(ws::GlobalWorkspace; init=false)
    summary(Base.stdout, ws, Val(init))
end

Base.summary(io::IO, ws::GlobalWorkspace; init=false) = summary(io, ws, Val(init))

function Base.summary(io::IO, ws::GlobalWorkspace, ::Val{true})
    println(io, "Number of MCMC iterations: ", num_mcmc_steps(ws))
    println(io, "Initial θ guess: ", state(ws))
    println(io, "Number of updates at each MCMC iteration: ", num_updt(ws))
    println(io, "Chosen `GlobalWorkspace:` ", remove_curly(typeof(ws)))
end

function Base.summary(io::IO, ws::GlobalWorkspace, ::Val{false})
    println(io, "Number of MCMC iterations: ", num_mcmc_steps(ws))
    println(io, "Estimated E[θ]: ", estim_mean(ws))
    println(io, "Estimated Cov[θ]: ")
    show(io, "text/plain", estim_cov(ws))
end

#                              LOCAL WORKSPACE
#-------------------------------------------------------------------------------

#=
        for custom LocalWorkspaces these methods MUST be overwritten:
                                                                              =#
"""
    create_workspace(
        ::MCMCBackend,
        mcmcupdate,
        global_ws::GlobalWorkspace,
        num_mcmc_steps
    ) where {T}

Create a local workspace for a given `mcmcupdate`.
"""
function create_workspace(
        ::MCMCBackend,
        mcmcupdate,
        global_ws::GlobalWorkspace,
        num_mcmc_steps
    ) where {T}
    error("create_workspace (localworkspace) is not defined.")
end

"""
    accepted(ws::LocalWorkspace, i::Int)

Return boolean for whether the ith update has been accepted.
"""
function accepted(ws::LocalWorkspace, i::Int)
    error("accepted(ws::LocalWorkspace, i) is not implemented")
end

"""
    set_accepted!(ws::LocalWorkspace, i::Int, v)

Set boolean for whether the ith update has been accepted.
"""
set_accepted!(ws::LocalWorkspace, i::Int, v) = (ws.acceptance_history[i] = v)

#=
        for custom LocalWorkspaces these methods will work so long as
        they contain sub_ws::StandardGlobalSubworkspace and
        sub_ws°::StandardGlobalSubworkspace, otherwise they need to be
        overwritten.
                                                                              =#
"""
    ll(ws::LocalWorkspace)

Return log-likelihood of the currently accepted parameter.
"""
ll(ws::LocalWorkspace) = ll(ws.sub_ws)

"""
    ll°(ws::LocalWorkspace)

Return log-likelihood of the currently proposed parameter.
"""
ll°(ws::LocalWorkspace) = ll(ws.sub_ws°)

"""
    ll(ws::LocalWorkspace, i::Int)

Return log-likelihood of the ith accepted parameter.
"""
ll(ws::LocalWorkspace, i::Int) = ll(ws.sub_ws, i)

"""
    ll°(ws::LocalWorkspace, i::Int)

Return log-likelihood of the ith proposed parameter.
"""
ll°(ws::LocalWorkspace, i::Int) = ll(ws.sub_ws°)

"""
    state(ws::LocalWorkspace)

Return the last accepted state.
"""
state(ws::LocalWorkspace) = state(ws.sub_ws)

"""
    state°(ws::LocalWorkspace)

Return the last proposed state.
"""
state°(ws::LocalWorkspace) = state(ws.sub_ws°)

#=
        for custom LocalWorkspaces these are well defined, but may
        not make sense:
                                                                              =#
"""
    create_workspaces(v::MCMCBackend, mcmc::MCMC)

Create local workspaces, one for each update.
"""
function create_workspaces(backend::MCMCBackend, mcmc::MCMC)
    [
        create_workspace(
            backend,
            mcmc.updates[i],
            mcmc.workspace,
            mcmc.schedule.num_mcmc_steps
        ) for i in 1:mcmc.schedule.num_updates
    ]
end

"""
    llr(ws::LocalWorkspace, i::Int)

Compute log-likelihood ratio at ith mcmc iteration.
"""
llr(ws::LocalWorkspace, i::Int) = sum(ll°(ws, i) .- ll(ws, i))

"""
    name_of_update(ws::LocalWorkspace)

Return the name of the update.
"""
name_of_update(ws::LocalWorkspace) = "unknown name"



#=
                        standard local sub-Workspace
                                                                              =#

"""
    struct StandardLocalSubworkspace{T} <: LocalWorkspace{T}
        state::Vector{T}
        ll::Vector{Float64}
        ll_history::Vector{Vector{Float64}}
        ∇ll::Vector{Vector{Float64}}
        momentum::Vector{Float64}
    end

Standard containers likely to be present in every local workspace. `state` is
the currently accepted *subset* of parameter θ that the corresponding update
operates on, `ll` is the corresponding log-likelihood and `ll_history` is the
chain of log-likelihoods. A single `ll` is of the type `Vector{Float64}` to
reflect the fact that a problem might admit a natural factorisation into
independent components that may be operated on independently, in parallel with
each entry in `ll` corresponding to a separate component. `∇ll` is the gradient
of log-likelihood (needed by gradient-based algorithms) and `momentum` is the
variable needed for the Hamiltionan dynamics. `∇ll` and `momentum` may be simply
left untouched if the problem does not need them.
"""
struct StandardLocalSubworkspace{T} <: LocalWorkspace{T}
    state::Vector{T}
    ll::Vector{Float64}
    ll_history::Vector{Vector{Float64}}
    ∇ll::Vector{Vector{Float64}}
    momentum::Vector{Float64}

    function StandardLocalSubworkspace(
            state::Vector{T}, M; num_indep_blocks=1
        ) where T
        new{T}(
            deepcopy(state),
            Float64[-Inf for _=1:num_indep_blocks],
            [zeros(Float64, num_indep_blocks) for _=1:M],
            [zeros(Float64, length(state)) for _=1:num_indep_blocks],
            zero(state),
        )
    end
end

ll(ws::StandardLocalSubworkspace) = ws.ll
ll(ws::StandardLocalSubworkspace, i::Int) = ws.ll_history[i]
state(ws::StandardLocalSubworkspace) = ws.state

#=
                        simple, custom LocalWorkspace
                                                                              =#

"""
    struct GenericLocalWorkspace{T} <: LocalWorkspace{T}
        sub_ws::StandardLocalSubworkspace{T}
        sub_ws°::StandardLocalSubworkspace{T}
        acceptance_history::Vector{Bool}
    end

Generic local workspace with `sub_ws` containing a subset of `state` that the
corresponding updates operates on, with currently accepted value of `state` as
well as its log-likelihood. `sub_ws°` corresponds to a proposal `state`.
`acceptance_history` keeps track of accept/reject decisions.
"""
struct GenericLocalWorkspace{T} <: LocalWorkspace{T}
    sub_ws::StandardLocalSubworkspace{T}
    sub_ws°::StandardLocalSubworkspace{T}
    acceptance_history::Vector{Bool}
    updt_name::String
end

function create_workspace(
        ::GenericMCMCBackend,
        mcmcupdate,
        global_ws::GenericGlobalWorkspace{T},
        num_mcmc_steps
    ) where {T}
    _state = state(global_ws, mcmcupdate)[:]
    GenericLocalWorkspace{T}(
        StandardLocalSubworkspace(_state, num_mcmc_steps),
        StandardLocalSubworkspace(_state, num_mcmc_steps),
        Vector{Bool}(undef, num_mcmc_steps),
        string(remove_curly(typeof(mcmcupdate))),
    )
end

accepted(ws::GenericLocalWorkspace, i::Int) = ws.acceptance_history[i]
name_of_update(ws::GenericLocalWorkspace) = ws.updt_name
