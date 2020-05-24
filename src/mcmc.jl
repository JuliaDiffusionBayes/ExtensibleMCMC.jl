#===============================================================================

        Defines the struct `MCMC` which is used to determine an MCMC
        sampling procedure.

===============================================================================#

"""
    mutable struct MCMC
        updates::Vector
        updates_and_decorators::Vector
        backend::MCMCBackend
        schedule::MCMCSchedule
        workspace::GlobalWorkspace
    end

A struct for defining a list of updates and specifying the backend algorithm
doing the MCMC sampling. Any other functionality is done solely by the internal
mechanizms of function the `run!`.

    MCMC(
        updt_and_decor::Vector{<:Union{MCMCUpdate,MCMCUpdateDecorator}};
        backend=GenericMCMCBackend()
    )

The main constructor of an `MCMC` struct. It accepts an array of updates and
update decorators `updates_and_decorators` which together constitute a single
MCMC step. Additionally, `backend` specialized to a particular, overarching MCMC
algorithm (if it exists). An example of a backend would be
`DiffusionMCMCBackend` (from the package `DiffusionMCMC.jl`).
"""
mutable struct MCMC
    updates::Vector
    updates_and_decorators::Vector
    backend::MCMCBackend
    schedule::MCMCSchedule
    workspace::GlobalWorkspace

    function MCMC(
            updt_and_decor::Vector{<:Union{MCMCUpdate,MCMCUpdateDecorator}};
            backend=GenericMCMCBackend()
        )
        new(
            strip_decorators(updt_and_decor),
            updt_and_decor,
            backend
        )
    end
end

"""
    strip_decorators(updt_and_decor)

Remove all decorators from a list
"""
strip_decorators(updt_and_decor) = filter(u->!isdecorator(u), updt_and_decor)

"""
    get_decorators(updt_and_decor)

Retrieve all decorators from a list
"""
get_decorators(updt_and_decor) = filter(u->isdecorator(u), updt_and_decor)

"""
    init!(
        mcmc::MCMC,
        num_mcmc_steps,
        data,
        θinit,
        exclude_updates=[];
        kwargs...
    )

Initialize the schedule and the global workspace of the MCMC sampler for a given
set of updates (already saved in `mcmc.updates`), a total number of MCMC
iterations given by `num_mcmc_steps`, observed dataset `data`. `θinit` is the
initial value of the main parameter that the MCMC sampling is done for,
`exclude_updates` lists the update indices and the repspective ranges of mcmc
iterations from which these updates are supposed to be omitted from and `kwargs`
lists all additional named arguments passed for creating a global workspace.
"""
function init!(
        mcmc::MCMC,
        num_mcmc_steps,
        data,
        θinit,
        exclude_updates=[];
        kwargs...
    )
    mcmc.workspace = init_global_workspace(
        mcmc.backend,
        num_mcmc_steps,
        mcmc.updates,
        data,
        θinit;
        kwargs...
    )
    mcmc.schedule = MCMCSchedule(
        num_mcmc_steps,
        length(mcmc.updates),
        exclude_updates;
        extra_schedule_params(
            mcmc.workspace,
            mcmc.updates_and_decorators;
            kwargs...
        )...
    )
end

function extra_schedule_params(
        workspace::GlobalWorkspace,
        updates_and_decorators;
        kwargs...
    )
    tuple()
end
