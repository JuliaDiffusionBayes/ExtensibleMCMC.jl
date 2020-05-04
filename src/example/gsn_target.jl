mutable struct GsnTargetLaw{T}
    θ::Vector{Float64}
    P::T

    function GsnTargetLaw(μ, Σ=Matrix{Float64}(I, (length(μ), length(μ))))
        d = length(μ)
        θ = zeros(d*(d+1))
        θ[1:d] .= μ
        θ[d+1:d*d+d] .= vec(Σ)
        P = MvNormal(μ, Σ)
        new{typeof(P)}(θ, P)
    end
end

function set_parameters!(P::GsnTargetLaw, loc2glob_idx, θ)
    P.θ[loc2glob_idx] .= θ
    d = dim(P.P.Σ)
    μ, Σ = P.θ[1:d], reshape(P.θ[(d+1):end], (d,d))
    Σ = Symmetric(triu(Σ))
    P.P = MvNormal(μ, Σ)
end

function Distributions.loglikelihood(P::GsnTargetLaw, observs)
    ll = 0.0
    for obs in observs
        ll += logpdf(P.P, obs)
    end
    ll
end
