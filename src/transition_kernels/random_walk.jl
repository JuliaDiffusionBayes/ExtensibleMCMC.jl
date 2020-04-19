#===============================================================================

    Various random walkers, including:
        ✔ uni- and multivariate Uniform random walk
        ✗ uni- and multivariate Gaussian random walk
        ✗ mixture of two Gaussian random walks
    All of the above allow for incorporating a restriction on the positivity
    of any chosen subset of parameter's coordinates.

===============================================================================#

"""
    RandomWalk <: TransitionKernel

Supertype for all random walkers.
"""
abstract type RandomWalk <: TransitionKernel end

#===============================================================================
                            Uniform random walk
===============================================================================#

"""
    mutable struct UniformRandomWalk{T,S} <: RandomWalk
        ϵ::T
        pos::S
    end

A uniform random walker that moves all unrestricted coordinates according to the
additive rule xᵢ+U, with U∼Unif(-ϵᵢ,ϵᵢ) and moves all coordinates restricted to
be positive according to the rule xᵢexp(U), with U∼Unif(-ϵᵢ,ϵᵢ).
"""
mutable struct UniformRandomWalk{T,S} <: RandomWalk
    ϵ::T
    pos::S

    function UniformRandomWalk(ϵ, pos=custom_zero(ϵ, Bool))
        @assert all(ϵ .> 0.0)
        new{typeof(ϵ),typeof(pos)}(ϵ, pos)
    end
end

Base.eltype(::UniformRandomWalk{T}) where T = eltype(T)
Base.length(rw::UniformRandomWalk) = length(rw.ϵ)

"""
    Base.rand(rw::UniformRandomWalk, θ)

Sample a new state for a random walker `rw` that is currently in a state `θ`.
"""
Base.rand(rw::UniformRandomWalk, θ) = rand(Random.GLOBAL_RNG, rw, θ)

function Base.rand(
        rng::Random.AbstractRNG,
        rw::UniformRandomWalk,
        θ
    )
    _rand(u) = rand(rng, u)
    U = _rand.(Uniform.(-rw.ϵ, rw.ϵ))
    θ.* ( exp.(U).*rw.pos .+ 1.0.*.!rw.pos ) .+ U.*.!rw.pos
end


function Random.rand!(rng::Random.AbstractRNG, rw::UniformRandomWalk, θ, θ°)
    θ° .= rand(rng, rw, θ)
end

Random.rand!(rw::UniformRandomWalk, θ, θ°) = rand!(Random.GLOBAL_RNG, rw, θ, θ°)

"""
    Distributions.logpdf(rw::UniformRandomWalk, θ, θ°)

Evaluate the log-probability density function for a move of a random walker
from θ to θ°.
"""
function Distributions.logpdf(rw::UniformRandomWalk, θ, θ°)
    mapreduce(
        i->( rw.pos[i] ? -log(2.0*rw.ϵ[i])-log(θ°[i]) : 0.0 ),
        +,
        1:length(rw.ϵ)
    )
end


#===============================================================================
                            Gaussian random walk
===============================================================================#
