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
    ll::Vector{Float64}
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
        Float64[-Inf],
        deepcopy(data.P),
        deepcopy(data.P),
    )
end

#                              LOCAL WORKSPACE
#-------------------------------------------------------------------------------
"""
    struct StandardLocalSubworkspace{T} <: LocalWorkspace{T}
        state::Vector{T}
        ll::Vector{Float64}
        ll_history::Vector{Float64}
    end

Standard containers likely to be present in every local workspace. `state` is
the currently accepted *subset* of parameter θ that the corresponding update
operates on, `ll` is the corresponding log-likelihood and `ll_history` is the
chain of log-likelihoods.
"""
struct StandardLocalSubworkspace{T} <: LocalWorkspace{T}
    state::Vector{T}
    ll::Vector{Float64}
    ll_history::Vector{Float64}

    function StandardLocalSubworkspace(state::Vector{T}, M) where T
        new{T}(deepcopy(state), Float64[-Inf], Vector{Float64}(undef, M))
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
end

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
    )
end
