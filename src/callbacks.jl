#===============================================================================

    Implementations of callbacks---pieces of code that are run in-between
    MCMC updates. The following callbacks are implemented here:
        ✔   Saving callback: saves the history of the updated chain as a
            CSV file.
        ✗   Plotting callback: for online plotting of the Markov chains,
            acceptance rates, sampled paths etc. as they are evolving.
        ✗   REPL feedback callback: for printing progress messages to REPL.

===============================================================================#

"""
    Callback

Supertype of all callbacks. They specify additional actions that can be
performed before or after each MCMC step or at the very end of sampling.
"""
abstract type Callback end


#NOTE by default nothing to be done, but for instance a plotting callback will
# need to set up the canvas
"""
    init!(::Callback)

Initialization actions for a callback. By default nothing to be done.
"""
function init!(::Callback, ws::GlobalWorkspace) end

function init!(vec_of_callbacks::Vector{<:Callback}, ws::GlobalWorkspace)
    for v in vec_of_callbacks
        init!(v, ws)
    end
end

function update!(
        callbacks::Vector{<:Callback},
        ws::LocalWorkspace,
        global_ws::GlobalWorkspace,
        step
    )
    for callback in callbacks
        check_if_execute(callback, step) && execute!(callback, ws, global_ws, step)
    end
end



"""
    check_if_execute(callback::Callback, iter)

Check if `callback` is supposed to be executed at the mcmc iteration `iter`.
"""
check_if_execute(callback::Callback, iter) = false

"""
    execute!(callback::Callback, ws::GlobalWorkspace, iter)

Execute the `callback`.
"""
function execute!(callback::Callback, ws::LocalWorkspace, gws::GlobalWorkspace, iter) end

"""
    cleanup!(callback::Callback, ws::GlobalWorkspace, iter)

The last call to callback, after all MCMC steps and before exiting the function
`run`.
"""
function cleanup!(callback::Callback, ws::GlobalWorkspace, iter) end


#===============================================================================
                            Saving callback
===============================================================================#
"""
    struct SavingCallback <: Callback
        save_at_the_end::Bool
        save_intermediate::Bool
        save_at_iters::Vector{Int64}
        filename::String
    end

Struct for saving the intermediate or final states of the sampled chain to the
disk.

        SavingCallback(;
            save_at_the_end=true,
            save_at_iters=[],
            overwrite]=false,
            filename="mcmc_results",
            add_datestamp=false,
            path=".",
        )

    The main constructor for `SavingCallback`. `save_at_the_end` is an indicator
    flag for whether to save to a file at the end of mcmc sampling,
    `save_at_iters` additionally specifies at which intermediate iterations
    the chain should be saved to a file, `overwrite` if set to true will
    overwrite any already existing file that shares the name with the one passed
    to this function. `filename` is the main stem of the file's name,
    `add_datestamps` will add the date and time at the time of creating the
    callback to the filename and `path` specifies the directory path to save to.
"""
struct SavingCallback <: Callback
    save_at_the_end::Bool
    save_intermediate::Bool
    save_at_iters::Vector{Int64}
    filename::String

    function SavingCallback(;
            save_at_the_end=true,
            save_at_iters=[],
            overwrite_at_save=false,
            filename="mcmc_results",
            add_datestamp=false,
            path=".",
        )
        timestamp = ( add_datestamp ? String("_", DateTime(Dates.now())) : "" )
        filename = String(filename, timestamp)
        full_filename = (
            !overwrite ?
            find_available_name(path, filename, "") :
            joinpath(path, String( filename, ".csv" ))
        )

        new(
            save_at_the_end,
            length(save_at_iters)>0,
            save_at_iters,
            full_filename,
        )
    end
end

"""
    find_available_name(path, filename, disambig_num, extension=".csv")

Check if a chosen filename already exists, if so, then tries appending
consecutive numbers to the end of the file until the first one that is not used
yet is found.
"""
function find_available_name(path, filename, disambig_num, extension=".csv")
    proposal_name = joinpath(path, String( filename, disambig_num, extension ))
    next_num = disambig_num == "" ? "1" : String(parse(Int64, disambig_num) + 1)
    isfile(proposal_name) && (
        return find_available_name(path, filename, next_num, extension)
    )
    proposal_name
end


"""
    init!(callback::SavingCallback)

Create a CSV file for writing into.
"""
init!(callback::SavingCallback, ws::GlobalWorkspace) = (open(callback.filename, "w") do _ end)

"""
    check_if_execute(callback::SavingCallback, iter)

Return `true` if the mcmc iteration `iter` is one at which an intermediate save
is to be made.
"""
function check_if_execute(callback::SavingCallback, iter)
    !callback.save_intermediate && return false
    iter in callback.save_at_iters && return true
    false
end

"""
    cleanup!(callback::SavingCallback, ws::GlobalWorkspace, iter)

Save the entire MCMC chain, history of proposals, acceptance history etc.
"""
function cleanup!(callback::SavingCallback, ws::GlobalWorkspace, iter)
    callback.save_at_the_end && execute!(callback, ws, iter)
end


"""
    execute!(sc::SavingCallback, ws::GlobalWorkspace, iter)

Save the chain of accepted states, proposed states and acceptance history to
the disk.
"""
function execute!(sc::SavingCallback, lws::LocalWorkspace, ws::GlobalWorkspace, iter)
    iter_start = find_starting_idx(sc, iter)
    open(sc.filename, "a") do f
        for i in iter_start:(iter-1)
            data_to_csv(f, ws, i)
        end
    end
end

"""
    find_starting_idx(callback::SavingCallback, iter)

Find the last index for which saving was done.
"""
function find_starting_idx(callback::SavingCallback, iter)
    !callback.save_intermediate && return 1
    iter_idx = first(searchsorted(callback.save_at_iters, iter))
    callback.save_at_iters[iter_idx-1]
end

"""
    data_to_csv(f, ws::GlobalWorkspace, i)

Write data entries to a CSV file.
"""
function data_to_csv(f, ws::GlobalWorkspace, i)
    for j in 1:length(ws.state_history[i])
        idx = "$i, $j, "
        θ = join(["$θₖ, " for θₖ in ws.state_history[i][j]])
        θ° = join(["$θₖ, " for θₖ in ws.state_proposal_history[i][j]])
        a_r = acceptance_history[i][j] ? 1 : 0
        write(f, String(idx, θ, θ°, a_r, "\n"))
    end
end


#===============================================================================
                        REPL progress printing callback
===============================================================================#
struct ProgressPrintCallback <: Callback

end
