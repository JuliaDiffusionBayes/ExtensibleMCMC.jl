#===============================================================================
    Definitions of abstract types reflecting conceptual inheritence structure
===============================================================================#

"""
Supertype of all MCMC updates
"""
abstract type MCMCUpdate end

"""
    MCMCParamUpdate <: MCMCUpdate

Supertype of all updates that make changes to the main MCMC parameter called
`state` in the `GlobalWorkspace`.
"""
abstract type MCMCParamUpdate <: MCMCUpdate end

"""
    MCMCGradientBasedUpdate <: MCMCParamUpdate

Supertype of all updates that require local gradient information to perform
update of `state`.
"""
abstract type MCMCGradientBasedUpdate <: MCMCParamUpdate end

"""
    MCMCConjugateUpdate <: MCMCUpdate

Supertype of all conjugate updates, for which sampling can be done directly,
without resorting to a Metropolis-Hastings algorithm
"""
abstract type MCMCConjugateUpdate <: MCMCUpdate end

"""
    MCMCConjugateParamUpdate <: MCMCParamUpdate

Supertype of all conjugate updates that update parameters and for which sampling
can be done directly, without resorting to a Metropolis-Hastings algorithm
"""
abstract type MCMCConjugateParamUpdate <: MCMCParamUpdate end

"""
    MCMCImputation <: MCMCUpdate

Supertype of all updates that do not make any changes to the main parameter
`state` in `GlobalWorkspace`, but instead, perform sampling on any auxiliary
variables that are not of direct interest to the MCMC chain.
"""
abstract type MCMCImputation <: MCMCUpdate end

"""
Supertype of all decorators to update schemes. In this context, we use the word
`decorator` to refer to any additional information that needs to be conveyed to
the MCMC sampler that is not directly extractable from the update-objects
themselves (for instance, a change in the delimitation of blocks that is made
in-between updates).
"""
abstract type MCMCUpdateDecorator end

"""
    isdecorator(u)

Returns true if `u` is a subtype of decorators.
"""
isdecorator(u) = (typeof(u) <: MCMCUpdateDecorator)

"""
Supertype of all workspaces—i.e. of structs that gather in one place various
objects that the MCMC sampler operates on.
"""
abstract type Workspace end

"""
    GlobalWorkspace{T} <: Workspace

Supertype of all global workspaces. Each MCMC sampler must have a unique global
workspace, which contains `state`, `state_history`, `state_proposal_history`,
`acceptance_history` and `data`. `state` is the paramater vector that the MCMC
sampling is done for (other names being self-explanatory).
"""
abstract type GlobalWorkspace{T} <: Workspace end

"""
    LocalWorkspace{T} <: Workspace

Supertype of all local workspaces. Local workspace should contain any additional
gathering of objects that are needed by specific updates, but are not are not
already in a global workspace. Each MCMC update has its own `LocalWorkspace`.
"""
abstract type LocalWorkspace{T} <: Workspace end

"""
Supertype of all transition kernels that perform updates on the main `state`
of the global workspace. Conceptually, these are used by subtypes of
`MCMCParamUpdate` to perform actual sampling and instances of types inheriting
from `TransitionKernel` are usually member objects of the instances inheriting
from `MCMCParamUpdate`.
"""
abstract type TransitionKernel end

"""
Supertype for all adaptation schemes.
"""
abstract type Adaptation end


"""
Supertype of all backends for the MCMC sampler.
"""
abstract type MCMCBackend end

"""
    GenericMCMCBackend <: MCMCBackend

A flag that no specific backend is passed.
"""
struct GenericMCMCBackend <: MCMCBackend end


abstract type ChainStats end

struct Previous end
struct Proposal end
struct PreMCMCStep end
struct PostMCMCStep end

#NOTE also appears in DiffusionDefinition.jl, but not used there.
"""
    remove_curly(::Type{K}) where K

Utility function that removes all type-specifiers listed in the curly brackets.

# Examples
```julia-repl
julia> remove_curly(Array{Float64,1})
Array
```
"""
@generated function remove_curly(::Type{K}) where K
    name_without_curly = Meta.parse(string(K)).args[1]
    :( $name_without_curly )
end
