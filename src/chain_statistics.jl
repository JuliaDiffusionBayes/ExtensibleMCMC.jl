mutable struct GenericChainStats{T} <: ChainStats
    rolling_ar::Vector{Vector{Float64}}
    roll_window::Int64
    mean::Vector{T}
    cov::Matrix{T}
    N::Int64

    function GenericChainStats(state::Vector{T}, updt_len, num_mcmc_steps, roll_window=100) where T
        new{T}(
            [zeros(Float64, updt_len) for _=1:num_mcmc_steps],
            roll_window,
            zero(state),
            zeros(T, (length(state), length(state))),
            1
        )
    end

end


function update!(cs::GenericChainStats, ws, step)
    old_sum_sq = (cs.N-1)/cs.N * cs.cov + cs.mean * cs.mean'
    cs.mean .= cs.mean .* (cs.N/(cs.N+1)) .+ ws.state ./ (cs.N+1)
    new_sum_sq = old_sum_sq + (ws.state * ws.state')/cs.N
    cs.cov .= new_sum_sq - (cs.N+1)/cs.N*(cs.mean * cs.mean')


    accepted_now = ws.acceptance_history[step.mcmciter][step.pidx]
    ra_prev = cs.rolling_ar[max(1,step.mcmciter-1)][step.pidx]
    accepted_expired = (
        step.mcmciter > cs.roll_window ?
        ws.acceptance_history[step.mcmciter-cs.roll_window][step.pidx] :
        false
    )
    cs.rolling_ar[step.mcmciter][step.pidx] = (
        ra_prev
        + (accepted_now - accepted_expired)/cs.roll_window
    )
    cs.N += 1
end
