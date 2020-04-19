"""
    mutable struct MCMCSchedule
        num_mcmc_steps::Int64
        num_updates::Int64
        start::NamedTuple{(:mcmciter, :pidx),Tuple{Int64,Int64}}
        exclude_updates::DefaultDict{Int64,OrdinalRange{Int64,Int64},UnitRange{Int64}}
    end

An object used for iterating over the steps of MCMC samplers. `num_mcmc_steps`
is the total number of MCMC steps, whereas `num_updates` is a total number of
separate updates that can be performed in a single MCMC step. Not all of those
updates have to be performed and the pattern of excluding the updates is stored
in `exclude_updates`. The `num_updates` and `exclude_updates` can change in the
midst of iterating through the `MCMCSchedule` and the iterator will register
that and act accordingly.
"""
mutable struct MCMCSchedule
    num_mcmc_steps::Int64
    num_updates::Int64
    start::NamedTuple{(:mcmciter, :pidx),Tuple{Int64,Int64}}
    exclude_updates::DefaultDict{Int64,OrdinalRange{Int64,Int64},UnitRange{Int64}}

    function MCMCSchedule(
            num_mcmc_steps,
            num_updates,
            exclude_updates=[]
        ) where {K,S}
        tuples_excluded_updates = DefaultDict{Int64,OrdinalRange{Int64,Int64}}(0:0)
        for exclude_update in exclude_updates
            for idx in exclude_update[1]
                tuples_excluded_updates[idx] = exclude_update[2]
            end
        end
        new(
            num_mcmc_steps,
            num_updates,
            (mcmciter=1, pidx=1),
            tuples_excluded_updates
        )
    end
end

"""
    Base.iterate(iter::MCMCSchedule, state=iter.start)

Iterate through the `MCMCSchedule`, outputting the named tuples
`(mcmciter=..., pidx=...)` along the way, indicating the current index of the
mcmc sampler (`mcmciter`), as well as the index of update that is to be
performed.
"""
function Base.iterate(iter::MCMCSchedule, state=iter.start)
    state.mcmciter > iter.num_mcmc_steps && return nothing
    return (state, transition(iter, state))
end


"""
    transition(schedule::MCMCSchedule, state)

Determine the next state of the `MCMCSchedule` iterator, by skipping through
all updates that are to be excluded, as per `schedule.exclude_updates`.
"""
function transition(schedule::MCMCSchedule, state)
    pidx_to_reset = (state.pidx == schedule.num_updates)
    new_state = (
        mcmciter = state.mcmciter + pidx_to_reset,
        pidx = ( pidx_to_reset ? 1 : state.pidx + 1 ),
    )
    iterations_to_skip = schedule.exclude_updates[new_state.pidx]
    skip_state = new_state.mcmciter in iterations_to_skip
    skip_state && return transition(schedule, new_state)
    new_state
end

"""
    reschedule!(
        schedule::MCMCSchedule,
        num_new_updates=0,
        idxes_to_remove=[],
        idxes_to_add=[]
    )

Make changes to the `schedule` (possibly in the midst of iterating through it).
Add `num_new_updtes`-many allowable updates that can be performed at each MCMC
step, `idxes_to_remove` lists all parameter udpates that are to be completely
removed from the MCMC sampler, `idxes_to_add` lists entries that are to be
added to `schedule.exclude_updates`.
"""
function reschedule!(
        schedule::MCMCSchedule,
        num_new_updates=0,
        idxes_to_remove=[],
        idxes_to_add=[]
    )
    schedule.num_updates += num_new_updates
    for idx in idxes_to_remove
        schedule.exclude_updates[idx] = 1:schedule.num_mcmc_steps
    end
    for idx_to_add in idxes_to_add
        schedule.exclude_updates[idx_to_add[1]] = idx_to_add[2]
    end
end
