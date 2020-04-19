#===============================================================================

    Adaptation schemes for adjusting parameter update steps. Contains
        ✔   Adaptation of uniform random walkers by changing ϵ to target the
            acceptance rate
        ✗   Haario-type adaptive schemes that adjusts the correlation of
            a Gaussian random walker to match that of the empirical covariance
        ✗   Adaptations for MALA algorithms, google search gives
            https://link.springer.com/article/10.1007/s11009-006-8550-0 (compare
            to STAN, decide accordingly)
        ✗   Adaptations for Hamiltonian Monte Carlo, google search gives
            http://proceedings.mlr.press/v28/wang13e.pdf (compare to STAN,
            decide accordingly)

===============================================================================#

#===============================================================================
                Adaptations for the Uniform Random Walker
===============================================================================#

"""
    NoAdaptation

A struct-flag for indicating that no adaptation is to be done.
"""
struct NoAdaptation <: Adaptation end

"""
    mutable struct AdaptationUnifRW{T} <: Adaptation
        proposed::Int64
        accepted::Int64
        target_accpt_rate::Float64
        adapt_every_k_steps::Int64
        scale::T
        min::T
        max::T
        offset::T
        N::Int64
    end

A struct containing information about the way in which to adapt the steps of the
uniform random walker. `proposed` and `accepted` are the internal counters that
keep track of the number of proposed and accepted samples. `target_accpt_rate`
is the acceptance rate of the Metropolis-Hastings steps that is supposed to be
targetted. `min` is used to enforce the minimal allowable range that the random
walker can sample from, `max` enforces the maximum. `offset` introduces some
delay in the start of decelerting the adaptation extent and `scale` is a scaling
parameter for adaptation speed. `N` is the number of coordinates of the random
walker.
"""
mutable struct AdaptationUnifRW{T} <: Adaptation
    proposed::Int64
    accepted::Int64
    target_accpt_rate::Float64
    adapt_every_k_steps::Int64
    scale::T
    min::T
    max::T
    offset::T
    N::Int64

    function AdaptationUnifRW{T}(
            target_accpt_rate, adapt_every_k_steps, scale::T,
            min, max, offset, N,
        ) where T
        new{T}(
            0, 0, target_accpt_rate, adapt_every_k_steps, scale,
            min, max, offset, N
        )
    end
end

"""
    AdaptationUnifRW(θ::K; kwargs...) where K

Define an adaptation scheme for a random walker that does updates on the
parameter of shape and type `θ`. The following named parameters can be specified
...
# Arguments
- `adapt_every_k_steps=100`: number of steps based on which adaptation happens
- `target_accpt_rate=0.234`: acceptance rate of MH step that is to be targetted
- `scale=1.0`: scaling of the adaptation
- `min=1e-12`: minimum allowable (half)-range of the uniform sampler
- `max=1e7`: maximum allowable (half)-range of the unifor sampler
- `offset=1e2`: number of adaptation steps after which the shrinking of adaptive
                steps is supposed to start.
...
"""
function AdaptationUnifRW(θ::K; kwargs...) where K
    non_vec_kwargs = [:adapt_every_k_steps, :target_accpt_rate]
    trgt_ar = _assure_scalar( get(kwargs, non_vec_kwargs[2], 0.234) )
    steps = _assure_scalar( get(kwargs, non_vec_kwargs[1], 100) )
    possible_vec_kwargs = filter(u->!(u.first in non_vec_kwargs), kwargs)

    ulengths = unique( map(x->length(x), values(possible_vec_kwargs)) )

    lul = length(ulengths)
    @assert lul <= 2

    cat = ( lul == 0 || maximum(ulengths) == 1 ? Val(:scalar) : Val(:nonscalar) )

    _AdaptationUnifRW(
        ismutable(K), cat, Val(length(θ)), trgt_ar, steps;
        possible_vec_kwargs...
    )
end

"""
    _adpt_rw_fill_defaults(f::Function)

Return a vector with set parameters, setting them according to the "recipe" `f`,
falling back on default values if necessary.
"""
function _adpt_rw_fill_defaults(f::Function)
    [
        f(s, v) for (s, v) in
        [(:scale, 1.0), (:min, 1e-12), (:max, 1e7), (:offset, 1e2)]
    ]
end

"""
    _adpt_rw_fill_defaults(; kwargs...)

Return a vector with set scalar parameters, falling back on default values if
necessary.
"""
function _adpt_rw_fill_defaults(; kwargs...)
    _adpt_rw_fill_defaults( (s,v)->first(get(kwargs, s, v)) )
end

"""
    _adpt_rw_fill_defaults(N; kwargs...)

Return a vector with set vector parameters, falling back on default values if
necessary.
"""
function _adpt_rw_fill_defaults(N; kwargs...)
    _adpt_rw_fill_defaults(
        (s,v)->_upgrade_to_vec( get(kwargs, s, repeat([v], N)), N )
    )
end

"""
    _adpt_rw_fill_defaults(N; kwargs...)

Return a vector with set `SVector` parameters, falling back on default values if
necessary.
"""
function _adpt_rw_fill_defaults(n::Val{N}; kwargs...) where N
    z = zero(SVector{N,Float64})
    _adpt_rw_fill_defaults(
        (s,v)->_upgrade_to_svec( get(kwargs, s, z.+v), n )
    )
end

"""
    _AdaptationUnifRW(
            ::Any, ::Val{:scalar}, ::Val{N}, trgt_ar, steps; kwargs...
        ) where N

Internal constructor of AdaptationUnifRW, populating entries with scalars.
"""
function _AdaptationUnifRW(
        ::Any, ::Val{:scalar}, ::Val{N}, trgt_ar, steps; kwargs...
    ) where N
    AdaptationUnifRW{Float64}(
        trgt_ar, steps, _adpt_rw_fill_defaults(; kwargs...)..., N
    )
end

"""
    _AdaptationUnifRW(
            ::Val{true}, ::Val{:nonscalar}, ::Val{N}, trgt_ar, steps; kwargs...
        ) where N

Internal constructor of AdaptationUnifRW, populating entries with vectors.
"""
function _AdaptationUnifRW(
        ::Val{true}, ::Val{:nonscalar}, ::Val{N}, trgt_ar, steps; kwargs...
    ) where N
    AdaptationUnifRW{Vector{Float64}}(
        trgt_ar, steps, _adpt_rw_fill_defaults(N; kwargs...)..., N
    )

end

"""
    _AdaptationUnifRW(
            ::Val{false}, ::Val{:nonscalar}, v::Val{N}, trgt_ar, steps; kwargs...
        ) where N

Internal constructor of AdaptationUnifRW, populating entries with `SVector`s.
"""
function _AdaptationUnifRW(
        ::Val{false}, ::Val{:nonscalar}, v::Val{N}, trgt_ar, steps; kwargs...
    ) where N
    AdaptationUnifRW{SVector{N,Float64}}(
        trgt_ar, steps, _adpt_rw_fill_defaults(v; kwargs...)..., N
    )
end

"""
    Base.:(==)(a::AdaptationUnifRW{T}, b::AdaptationUnifRW{S}) where{T,S}

Convenience comparison of `AdaptationUnifRW` instances. Used only for tests.
"""
function Base.:(==)(a::AdaptationUnifRW{T}, b::AdaptationUnifRW{S}) where{T,S}
    T != S && return false
    fns = fieldnames(AdaptationUnifRW)
    for fn in fns
        getfield(a, fn) != getfield(b, fn) && return false
    end
    true
end

"""
    isequal_except(
        ::AdaptationUnifRW{T}, b::AdaptationUnifRW{S}, args...
        ) where{T,S}

Convenience comparison of `AdaptationUnifRW` instances. Used only for tests.
The fields specified by `args` are excluded from comparison.
"""
function isequal_except(
        a::AdaptationUnifRW{T},
        b::AdaptationUnifRW{S},
        args...
    ) where{T,S}
    T != S && return false
    fns = fieldnames(AdaptationUnifRW)
    for fn in fns
        fn in args && (continue)
        getfield(a, fn) != getfield(b, fn) && return false
    end
    true
end

"""
    acceptance_rate(adpt::AdaptationUnifRW)

Compute current acceptance rate of the Metropolis-Hastings update step
"""
acceptance_rate(adpt::AdaptationUnifRW) = (
    (adpt.proposed == 0) ? 0.0 : adpt.accepted/adpt.proposed
)

"""
    acceptance_rate!(adpt::AdaptationUnifRW)

Destructive computation of a current acceptance rate that also resets the
number of proposals and accepted samples to zeros.
"""
function acceptance_rate!(adpt::AdaptationUnifRW)
    a_r = acceptance_rate(adpt)
    reset!(adpt)
    a_r
end

"""
    reset!(adpt::AdaptationUnifRW)

Reset the number of proposals and accepted samples to zero.
"""
function DataStructures.reset!(adpt::AdaptationUnifRW)
    adpt.proposed = 0
    adpt.accepted = 0
end

"""
    readjust!(rw::UniformRandomWalk, adpt::AdaptationUnifRW, mcmc_iter)

Adaptive readjustment of the range for sampling uniforms by the random walker.
"""
function readjust!(rw::UniformRandomWalk, adpt::AdaptationUnifRW, mcmc_iter)
    δ = compute_δ(adpt, mcmc_iter)
    a_r = acceptance_rate!(adpt)
    ϵ = compute_ϵ(rw.ϵ, param, a_r, δ)
    rw.ϵ = ϵ
    ϵ
end

"""
    register!(adpt::AdaptationUnifRW, accepted::Bool, ::Any)

Register the result of the acceptance decision in the Metropolis-Hastings step.
"""
function register!(adpt::AdaptationUnifRW, accepted::Bool, ::Any)
    adpt.accepted += accepted
    adpt.proposed += 1
end

"""
    compute_δ(p, mcmc_iter)

δ decreases roughly proportional to scale/sqrt(iteration)
"""
function compute_δ(p, mcmc_iter)
    p.scale/sqrt(
        max(
            1.0,
            mcmc_iter/p.adapt_every_k_steps-p.offset
        )
    )
end

"""
    compute_ϵ(ϵ_old, p, a_r, δ, flip=1.0, f=identity, finv=identity)

ϵ is moved by δ to adapt to target acceptance rate
"""
function compute_ϵ(ϵ_old, p, a_r, δ, flip=1.0, f=identity, finv=identity)
    ϵ = finv(f(ϵ_old) + flip*(2*(a_r > p.target_accpt_rate)-1)*δ)
    ϵ = max(min(ϵ,  p.max), p.min)    # trim excessive updates
end

#===============================================================================
                        Haario-type adaptive scheme
===============================================================================#

mutable struct HaarioTypeAdaptation <: Adaptation

end
