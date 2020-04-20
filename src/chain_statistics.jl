#===============================================================================
            Some simple online statistics for the Markov chain
===============================================================================#

"""
    mutable struct GenericChainStats{T} <: ChainStats
        rolling_ar::Vector{Vector{Float64}}
        roll_window::Int64
        mean::Vector{T}
        cov::Matrix{T}
        N::Int64
    end

Simple online statistics for the Markov chain.
"""
mutable struct GenericChainStats{T} <: ChainStats
    rolling_ar::Vector{Vector{Float64}}
    roll_window::Int64
    mean::Vector{T}
    cov::Matrix{T}
    N::Int64

    function GenericChainStats(
            state::Vector{T},
            updt_len,
            num_mcmc_steps,
            roll_window=100
        ) where T
        new{T}(
            [zeros(Float64, updt_len) for _=1:num_mcmc_steps],
            roll_window,
            zero(state),
            zeros(T, (length(state), length(state))),
            1
        )
    end

end


function update!(
        cs::GenericChainStats,
        global_ws::GlobalWorkspace,
        local_ws::LocalWorkspace,
        step
    )
    θ = global_ws.sub_ws.state
    old_sum_sq = (cs.N-1)/cs.N * cs.cov + cs.mean * cs.mean'
    cs.mean .= cs.mean .* (cs.N/(cs.N+1)) .+ θ ./ (cs.N+1)
    new_sum_sq = old_sum_sq + (θ * θ')/cs.N
    cs.cov .= new_sum_sq - (cs.N+1)/cs.N*(cs.mean * cs.mean')

    accepted_now = local_ws.acceptance_history[step.mcmciter]
    ra_prev = cs.rolling_ar[max(1,step.mcmciter-1)][step.pidx]
    accepted_outside_window = (
        step.mcmciter > cs.roll_window ?
        local_ws.acceptance_history[step.mcmciter-cs.roll_window] :
        false
    )
    cs.rolling_ar[step.mcmciter][step.pidx] = (
        ra_prev * cs.roll_window
        + (accepted_now - accepted_outside_window)
    )  / min(cs.roll_window, cs.N)
    cs.N += 1
end
