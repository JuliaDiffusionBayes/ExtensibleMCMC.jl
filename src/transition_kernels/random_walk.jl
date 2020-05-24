#===============================================================================

    Various random walkers, including:
        ✔ uni- and multivariate Uniform random walk
        ✔ multivariate Gaussian random walk
        ✔ mixture of two Gaussian random walks
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

@doc raw"""
    mutable struct UniformRandomWalk{T,S} <: RandomWalk
        ϵ::T
        pos::S
    end

A uniform random walker that moves all **unrestricted** coordinates according to
the additive rule
```math
x_i+U,
    \quad\mbox{with}
        \quad U\sim\texttt{Unif}([-ϵ_i,ϵ_i]),
            \qquad i\in\texttt{indices},
```
and moves all coordinates **restricted to be positive** according to the rule
```math
x_i\exp(U),
    \quad\mbox{with}
        \quad U\sim\texttt{Unif}([-ϵ_i,ϵ_i]),
            \qquad i\in\texttt{indices}.
```
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
@doc raw"""
    mutable struct GaussianRandomWalk{T} <: RandomWalk
        Σ::Symmetric{T,Array{T,2}}
        pos::Vector{Bool}
    end

A Gaussian random walker that first transforms all coordinates restricted to be
positive to a log-scale via
```math
θ_i\leftarrow\log(θ_i),\quad i\in\{i: θ_i\texttt{_restricted_to_be_positive}\},
```
then moves according to
```math
θ°∼N(θ, Σ),\quad \forall i,
```
and finally, transforms the restricted coordinates back to the original scale
via
```math
θ°_i\leftarrow\exp(θ°_i),\quad i\in\{i: θ_i\texttt{_restricted_to_be_positive}\}.
```
Restrictions are specified in the vector `pos` (and no restrictions correspond
to all entries in `pos` being `false`).
"""
mutable struct GaussianRandomWalk{T} <: RandomWalk
    Σ::Symmetric{T,Array{T,2}}
    pos::Vector{Bool}

    function GaussianRandomWalk(Σ, pos=nothing)
        @assert size(Σ, 1) == size(Σ, 2)
        Σ = reshape(Σ, (size(Σ, 1), size(Σ, 2)))
        T = eltype(Σ)
        pos = (pos === nothing) ? fill(false, size(Σ, 1)) : pos
        new{T}(Symmetric(Σ), pos)
    end
end

Base.eltype(::GaussianRandomWalk{T}) where T = T
Base.length(rw::GaussianRandomWalk) = size(rw.Σ, 1)
remove_constraints!(rw::GaussianRandomWalk, θ) = (θ[rw.pos] .= log.(θ[rw.pos]))
reimpose_constraints!(rw::GaussianRandomWalk, θ) = (
    θ[rw.pos] .= exp.(θ[rw.pos])
)

Base.rand(rw::GaussianRandomWalk, θ) = rand(Random.GLOBAL_RNG, rw, θ)

function Base.rand(rng::Random.AbstractRNG, rw::GaussianRandomWalk, θ)
    remove_constraints!(rw, θ)
    θ° = rand(MvNormal(θ, rw.Σ))
    reimpose_constraints!(rw, θ°)
    reimpose_constraints!(rw, θ)
    θ°
end

function Random.rand!(rw::GaussianRandomWalk, θ, θ°)
    rand!(Random.GLOBAL_RNG, rw, θ, θ°)
end

function Random.rand!(rng::Random.AbstractRNG, rw::GaussianRandomWalk, θ, θ°)
    θ° .= rand(rng, rw, θ)
end

_logjacobian(rw::GaussianRandomWalk, θ) = -sum(log.(θ[rw.pos]))

function Distributions.logpdf(rw::GaussianRandomWalk, θ, θ°)
    logJ = _logjacobian(rw, θ°)
    remove_constraints!(rw, θ)
    remove_constraints!(rw, θ°)
    lpdf = logpdf(MvNormal(θ, rw.Σ), θ°) + logJ
    reimpose_constraints!(rw, θ)
    reimpose_constraints!(rw, θ°)
    lpdf
end

#===============================================================================
            A mixture of two multidimensional Gaussian random walks
===============================================================================#
@doc raw"""
    mutable struct GaussianRandomWalkMix{T} <: RandomWalk
        gsn_A::GaussianRandomWalk{T}
        gsn_B::GaussianRandomWalk{T}
        λ::Float64
    end

A mixture of two Gaussian random walkers. It performs updates according to
```math
X_{i+1} = X_i + λZ_A + (1-λ)Z_B,
```
where
```math
Z_A ∼ N(0,Σ_A),\quad\mbox{and}\quad Z_B ∼ N(0, Σ_B),
```
(with $X_i$ being already appropriately log-transformed if needed).
"""
mutable struct GaussianRandomWalkMix{T} <: RandomWalk
    gsn_A::GaussianRandomWalk{T}
    gsn_B::GaussianRandomWalk{T}
    λ::Float64

    function GaussianRandomWalkMix(Σ_A, Σ_B, λ=0.5, pos=nothing)
        @assert 0.0 <= λ <= 1.0
        @assert size(Σ_A) == size(Σ_B)
        gsn_A = GaussianRandomWalk(Σ_A, pos)
        gsn_B = GaussianRandomWalk(Σ_B, pos)
        T = eltype(gsn_A)
        new{T}(gsn_A, gsn_B, λ)
    end
end

Base.eltype(::GaussianRandomWalkMix{T}) where T = T
Base.length(rw::GaussianRandomWalkMix) = length(rw.gsn_A)

Base.rand(rw::GaussianRandomWalkMix, θ) = rand(Random.GLOBAL_RNG, rw, θ)

function Base.rand(rng::Random.AbstractRNG, rw::GaussianRandomWalkMix, θ)
    rw_i = pick_kernel(rng, rw)
    rand(rng, rw_i, θ)
end

Random.rand!(rw::GaussianRandomWalkMix, θ, θ°) = rand!(Random.GLOBAL_RNG, rw, θ, θ°)

function Random.rand!(rng::Random.AbstractRNG, rw::GaussianRandomWalkMix, θ, θ°)
    rw_i = pick_kernel(rng, rw)
    rand!(rng, rw_i, θ, θ°)
end

function pick_kernel(rng::Random.AbstractRNG, rw::GaussianRandomWalkMix)
    rand(rng, Bernoulli(rw.λ)) ? rw.gsn_B : rw.gsn_A
end

function Distributions.logpdf(rw::GaussianRandomWalkMix, θ, θᵒ)
    log( (1-rw.λ)*exp(logpdf(rw.gsn_A, θ, θᵒ))
          + rw.λ *exp(logpdf(rw.gsn_B, θ, θᵒ)) )
end
