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

"""
    struct GenericGlobalWorkspace{T,SW,TL} <: GlobalWorkspace{T}
        sub_ws::SW
        ll::Vector{Float64}
        P::TL
        P°::TL
    end

Generic global workspace with `sub_ws` containing current `state` and keeping
track of its history and some basic statistics. `P` and `P°` are the target laws
with accepted and proposal `state` set as parameters.
"""
struct GenericGlobalWorkspace{T,SW,TL} <: GlobalWorkspace{T}
    sub_ws::SW
    P::TL
    P°::TL
end

"""
    init_global_workspace(
        ::GenericMCMCBackend,
        schedule::MCMCSchedule,
        updates::Vector{<:MCMCUpdate},
        data,
        θinit::Vector{T};
        kwargs...
    ) where T

Initialize the `GenericGlobalWorkspace`.
"""
function init_global_workspace(
        ::GenericMCMCBackend,
        schedule::MCMCSchedule,
        updates::Vector{<:MCMCUpdate},
        data,
        θinit::Vector{T};
        kwargs...
    ) where T
    sub_ws = StandardGlobalSubworkspace(
        schedule.num_mcmc_steps,
        length(updates),
        data,
        θinit,
    )
    GenericGlobalWorkspace{T,typeof(sub_ws),typeof(data.P)}(
        sub_ws,
        deepcopy(data.P),
        deepcopy(data.P),
    )
end

num_mcmc_steps(ws::GlobalWorkspace) = length(ws.sub_ws.state_history)
state(ws::GlobalWorkspace) = ws.sub_ws.state
num_updt(ws::GlobalWorkspace) = length(first(ws.sub_ws.state_history))
estim_mean(ws::GlobalWorkspace) = ws.sub_ws.stats.mean
estim_cov(ws::GlobalWorkspace) = ws.sub_ws.stats.cov

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
each entry in `ll` corresponding to a separate subcomponent. `∇ll` is the
gradient of log-likelihood (needed by gradient-based algorithms) and `momentum`
is the variable needed for the Hamiltionan dynamics. `∇ll` and `momentum` may be
simply left untouched if the problem does not need them.
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

accepted(ws::LocalWorkspace, i) = ws.acceptance_history[i]
ll(ws::LocalWorkspace, i) = ws.sub_ws.ll_history[i]
ll°(ws::LocalWorkspace, i) = ws.sub_ws°.ll_history[i]
ll_prop = ll°
llr(ws::LocalWorkspace, i) = sum(ll°(ws, i) .- ll(ws, i))
state(ws::LocalWorkspace) = ws.sub_ws.state
state°(ws::LocalWorkspace) = ws.sub_ws°.state
state_prop = state°
name_of_update(ws::LocalWorkspace) = ws.updt_name

"""
    create_workspaces(v::GenericMCMCBackend, mcmc::MCMC)

Create local workspaces, one for each update.
"""
function create_workspaces(v::GenericMCMCBackend, mcmc::MCMC)
    [
        create_workspace(
            v,
            mcmc.updates[i],
            mcmc.workspace,
            mcmc.schedule.num_mcmc_steps
        ) for i in 1:mcmc.schedule.num_updates
    ]
end


"""
    create_workspace(
        ::GenericMCMCBackend,
        mcmcupdate,
        global_ws::GenericGlobalWorkspace{T},
        num_mcmc_steps
    ) where {T}

Create a local workspace for a given `mcmcupdate`.
"""
function create_workspace(
        ::GenericMCMCBackend,
        mcmcupdate,
        global_ws::GenericGlobalWorkspace{T},
        num_mcmc_steps
    ) where {T}
    state = global_ws.sub_ws.state[mcmcupdate.loc2glob_idx]
    GenericLocalWorkspace{T}(
        StandardLocalSubworkspace(state, num_mcmc_steps),
        StandardLocalSubworkspace(state, num_mcmc_steps),
        Vector{Bool}(undef, num_mcmc_steps),
        string(remove_curly(typeof(mcmcupdate))),
    )
end
