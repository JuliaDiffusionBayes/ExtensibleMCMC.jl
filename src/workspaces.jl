struct GenericGlobalWorkspace{T,TD,TL} <: GlobalWorkspace{T}
    state::Vector{T}
    ll::Vector{Float64}
    ll°::Vector{Float64}
    state_history::Vector{Vector{Vector{T}}}
    state_proposal_history::Vector{Vector{Vector{T}}}
    ll_history::Vector{Vector{Float64}}
    ll°_history::Vector{Vector{Float64}}
    acceptance_history::Vector{Vector{Bool}}
    data::TD
    P::TL
    P°::TL
    stats::GenericChainStats{T}
end

function init_global_workspace(
        ::GenericMCMCBackend,
        schedule::MCMCSchedule,
        updates::Vector{<:MCMCUpdate},
        data,
        θinit::Vector{T};
        kwargs...
    ) where T
    M = schedule.num_mcmc_steps
    ∅s = map(u->Vector{T}(undef, length(θinit)), updates)
    GenericGlobalWorkspace{T,typeof(data),typeof(data.P)}(
        deepcopy(θinit),
        Float64[-Inf],
        Float64[-Inf],
        [deepcopy(∅s) for _=1:M],
        [deepcopy(∅s) for _=1:M],
        [Vector{Float64}(undef, length(updates)) for _=1:M],
        [Vector{Float64}(undef, length(updates)) for _=1:M],
        [Vector{Bool}(undef, length(updates)) for _=1:M],
        data,
        deepcopy(data.P),
        deepcopy(data.P),
        GenericChainStats(θinit, length(∅s), M),
    )
end

struct GenericLocalWorkspace{T} <: LocalWorkspace{T}
    state::Vector{T}
    state°::Vector{T}
    #ll::Vector{Float64}
    #ll°::Vector{Float64}
    #state_history::Vector{Vector{T}}
    #state_proposal_history::Vector{Vector{T}}
    #ll_history::Vector{Float64}
    #ll°_history::Vector{Float64}
    acceptance_history::Vector{Bool}
    #data::TD
end

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

function create_workspace(
        ::GenericMCMCBackend,
        mcmcupdate,
        global_ws::GenericGlobalWorkspace{T},
        num_mcmc_steps
    ) where {T}
    state = global_ws.state[mcmcupdate.loc2glob_idx]
    #∅ = Vector{T}(undef, length(mcmcupdate.loc2glob_idx))
    GenericLocalWorkspace{T}(
        deepcopy(state),
        deepcopy(state),
        #Float64[0.0],
        #Float64[0.0],
        #[deepcopy(∅) for _=1:num_mcmc_steps],
        #[deepcopy(∅) for _=1:num_mcmc_steps],
        #Vector{Float64}(undef, num_mcmc_steps),
        #Vector{Float64}(undef, num_mcmc_steps),
        Vector{Bool}(undef, num_mcmc_steps),
        #global_ws.data
    )
end
