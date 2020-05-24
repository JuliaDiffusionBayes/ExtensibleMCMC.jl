#===============================================================================

    Implementations of callbacks---pieces of code that are run in-between
    MCMC updates. The following callbacks are implemented here:
        ✔   Saving callback: saves the history of the updated chain as a
            CSV file.
        ✔   REPL feedback callback: for printing progress messages to REPL.

===============================================================================#

"""
Supertype of all callbacks. They specify additional actions that can be
performed before or after each MCMC step or at the very end of sampling.
"""
abstract type Callback end


#NOTE by default nothing to be done, but for instance a plotting callback will
# need to set up the canvas
"""
    init!(::Callback, ws::GlobalWorkspace)

Initialization actions for a callback. By default nothing to be done.
"""
function init!(::Callback, ws::GlobalWorkspace) end

function init!(callbacks::Vector{<:Callback}, ws::GlobalWorkspace)
    for c in callbacks
        init!(c, ws)
    end
end

function update_callbacks!(
        callbacks::Vector{<:Callback},
        global_ws::GlobalWorkspace,
        local_wss::Vector{<:LocalWorkspace},
        step,
        flag,
    )
    for callback in callbacks
        check_if_execute(callback, step, flag) && execute!(
            callback, global_ws, local_wss, step, flag
        )
    end
end



"""
    check_if_execute(callback::Callback, step, flag)

Check if `callback` is supposed to be executed at the mcmc step `step` with
pre- post- update flag `flag`.
"""
check_if_execute(callback::Callback, step, flag) = false

"""
    execute!(
        callback::Callback,
        global_ws::GlobalWorkspace,
        local_wss,
        step,
        flag
    )

Perform actions as specified by the `callback`. `local_wss` is a vector of all
local workspaces, `step` is an iterator from the `MCMCSchedule` and `flag` is
either `::PreMCMCStep` or `::PostMCMCStep`.
"""
function execute!(
        callback::Callback,
        global_ws::GlobalWorkspace,
        local_wss,
        step,
        flag
    )
end

"""
    cleanup!(callback::Callback, ws::GlobalWorkspace, step)

The last call to callback, after all MCMC steps and before exiting the function
`run!`.
"""
function cleanup!(callback::Callback, ws::GlobalWorkspace, step) end


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

Struct for saving the intermediate or final states of the sampled chain to a
hard drive.

    SavingCallback(;
        save_at_the_end=true,
        save_at_iters=[],
        overwrite_at_save=false,
        filename="mcmc_results",
        add_datestamp=false,
        path=".",
    )

The main constructor for `SavingCallback`.

# Arguments
- `save_at_the_end`: indicates whether to save to a file at the end of mcmc
                     sampling
- `save_at_iters`: specifies at which additional intermediate iterations the
                   chain should be saved to a file
- `overwrite`: set to true to overwrite any already existing file that shares
               the name with the one passed to this function
- `filename`: the main stem of the file's name
- `add_datestamps`: will add the date and time at the time of creating the
                    callback to the filename if set to `true`
- `path`: specifies the directory path to save to
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
        timestamp = (
            add_datestamp ?
            string(
                "_",
                Dates.format(
                    Dates.now(),
                    "yyyy-mm-dd_HH:MM:SS"
                )
            ) :
            ""
        )
        filename = string(filename, timestamp)
        full_filename = (
            !overwrite_at_save ?
            find_available_name(path, filename, "") :
            joinpath(path, string( filename, ".csv" ))
        )

        new(
            save_at_the_end,
            length(save_at_iters)>0,
            collect(save_at_iters),
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
    proposal_name = joinpath(path, string( filename, disambig_num, extension ))
    next_num = disambig_num == "" ? "1" : string(parse(Int64, disambig_num) + 1)
    isfile(proposal_name) && (
        return find_available_name(path, filename, next_num, extension)
    )
    proposal_name
end


"""
    init!(callback::SavingCallback)

Create a CSV file for writing into.
"""
function init!(callback::SavingCallback, ws::GlobalWorkspace)
    open(callback.filename, "w") do _ end
end

"""
    check_if_execute(callback::SavingCallback, iter)

Return `true` if the mcmc iteration `iter` is one at which an intermediate save
is to be made.
"""
function check_if_execute(callback::SavingCallback, step, ::PreMCMCStep)
    !callback.save_intermediate && return false
    step.mcmciter in callback.save_at_iters && step.pidx == 1 && return true
    false
end

"""
    cleanup!(callback::SavingCallback, ws::GlobalWorkspace, iter)

Save the entire MCMC chain, history of proposals, acceptance history etc.
"""
function cleanup!(callback::SavingCallback, ws::GlobalWorkspace, local_wss, step)
    callback.save_at_the_end && execute!(callback, ws, local_wss, step, nothing)
end


"""
    execute!(sc::SavingCallback, ws::GlobalWorkspace, iter)

Save the chain of accepted states, proposed states and acceptance history to
the disk.
"""
function execute!(sc::SavingCallback, ws::GlobalWorkspace, local_wss, step, ::Any)
    iter_start = find_starting_idx(sc, step)
    open(sc.filename, "a") do f
        for i in iter_start:(step.mcmciter-1)
            data_to_csv(f, ws, local_wss, i)
        end
    end
end

"""
    find_starting_idx(callback::SavingCallback, iter)

Find the last index for which saving was done.
"""
function find_starting_idx(callback::SavingCallback, step)
    !callback.save_intermediate && return 1
    iter_idx = first(searchsorted(callback.save_at_iters, step.mcmciter))
    iter_idx == 1 && return 1
    callback.save_at_iters[iter_idx-1]
end

"""
    data_to_csv(f, ws::GlobalWorkspace, i)

Write data entries to a CSV file.
"""
function data_to_csv(f, ws::GlobalWorkspace, local_wss, i)
    for j in 1:length(ws.sub_ws.state_history[i])
        idx = "$i, $j, "
        θ = join(["$θₖ, " for θₖ in ws.sub_ws.state_history[i][j]])
        θ° = join(["$θₖ, " for θₖ in ws.sub_ws.state_proposal_history[i][j]])
        ll = join(["$ll, " for ll in local_wss[j].sub_ws.ll_history[i]])
        ll° = join(["$ll, " for ll in local_wss[j].sub_ws°.ll_history[i]])
        a_r = join(["$a_r, " for a_r in local_wss[j].acceptance_history[i]])
        write(f, string(idx, "!, ", θ, "!, ", θ°, "!, ", ll, "!, ", ll°, "!,", a_r, "\n"))
    end
end


#===============================================================================
                        REPL progress printing callback
===============================================================================#
"""
    struct REPLCallback <: Callback
        print_every_k_iter::Int64
        show_all_updates::Bool
        basic_info_only::Bool
    end

Struct with instructions for printing progress messages to REPL.

    REPLCallback(;
        print_every_k_iter=100,
        show_all_upates=true,
        basic_info_only=true,
    )

Base constructor that uses named arguments.
"""
struct REPLCallback <: Callback
    print_every_k_iter::Int64
    show_all_updates::Bool
    basic_info_only::Bool

    function REPLCallback(;
            print_every_k_iter=100,
            show_all_upates=true,
            basic_info_only=true,
        )
        new(print_every_k_iter, show_all_upates, basic_info_only)
    end
end

function init!(callback::REPLCallback, ws::GlobalWorkspace)
    println(join(fill("*", 40)))
    println("Initializing an MCMC chain")
    summary(ws; init=true)
    println("* * *")
end

function check_if_execute(callback::REPLCallback, step, ::PostMCMCStep)
    (step.mcmciter % callback.print_every_k_iter == 0) || return false
    (callback.show_all_updates || step.pidx == 1) && return true
    false
end

function execute!(rc::REPLCallback, ws::GlobalWorkspace, local_wss, step, ::Any)
    lws, M = local_wss[step.pidx], step.mcmciter
    println("- - - - - - - - - - -")
    println(M, ".", step.pidx, " ", name_of_update(lws))
    _λ(x) = (length(x)==1 ? x[1] : x)
    _ll = map(x->round.(_λ(x(lws, M)), sigdigits=4), [ll, ll°, llr])
    a_r = accepted(lws, M) ? "✔" : "✗"
    println("\tll: $(_ll[1]), ll°: $(_ll[2]), llr: $(_ll[3]), a/r: $a_r")
    rc.basic_info_only || begin
        θs = map(x->round.(x(lws); sigdigits=4), [state, state°])
        println("\t\tθ : $(θs[1])")
        println("\t $a_r  θ°: $(θs[2])")
    end
end

function cleanup!(callback::REPLCallback, ws::GlobalWorkspace, local_wss, step)
    println("\n\nMCMC sampling has been successful!")
    println("Doing some clean-up and finishing...")
    println("\n⋆ ⋆ ⋆\n")
end
