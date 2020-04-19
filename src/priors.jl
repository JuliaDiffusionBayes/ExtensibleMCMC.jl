abstract type Prior end

"""
    ImproperPrior

Flat prior
"""
struct ImproperPrior <: Prior end
Distributions.logpdf(::ImproperPrior, θ) = 0.0

struct ImproperPosPrior <: Prior end
Distributions.logpdf(::ImproperPosPrior, θ) = -sum(log.(θ))

struct StandardPrior{T} <: Prior
    dist::T
    StandardPrior(dist::T) where T = new{T}(dist)
end
Distributions.logpdf(prior::StandardPrior, θ) = logpdf(prior.dist, θ)

struct ProductPrior{T,K} <: Prior
    dists::T
    idx::K

    function ProductPrior(dists, dims)
        dims_reformatted = Any[]
        last_idx = 1
        for dim in dims
            if dim == 1
                push!(dims_reformatted, dim)
                last_idx += 1
            else
                push!(dims_reformatted, last_idx:last_idx+dim-1)
                last_idx += dim
            end
        end
        idx = tuple(dims_reformatted...)
        dists = tuple(dists...)
        new{typeof(dists), typeof(idx)}(dists, idx)
    end
end

function Distributions.logpdf(prior::ProductPrior, θ)
    lp = 0.0
    for (dist, idx) in zip(prior.dists, prior.idx)
        lp += logpdf(dist, θ[idx])
    end
    lp
end
