"""
Supertype for all prior distributions
"""
abstract type Prior end

"""
    logpdf(::Prior, θ)

Evaluate the log-probability density function at `θ` for a given prior.
"""
function Distributions.logpdf(::T, θ) where {T<:Prior}
    error("logpdf not implemented for prior $T.")
end

"""
Flat prior
"""
struct ImproperPrior <: Prior end
Distributions.logpdf(::ImproperPrior, θ) = 0.0


"""
Flat prior for positive coordinates
"""
struct ImproperPosPrior <: Prior end
Distributions.logpdf(::ImproperPosPrior, θ) = -sum(log.(θ))

"""
    struct StandardPrior{T} <: Prior
        dist::T
    end

A standard prior π(θ).
"""
struct StandardPrior{T} <: Prior
    dist::T
    StandardPrior(dist::T) where T = new{T}(dist)
end
Distributions.logpdf(prior::StandardPrior, θ) = logpdf(prior.dist, θ)

@doc raw"""
    struct ProductPrior{T,K} <: Prior
        dists::T
        idx::K
    end

Generic prior distribution over parameter vector θ written in a facorized form.
For instance if the prior may be written as
```math
π(θ) = π_1(θ_{1:k})π_2(θ_{(k+1):n}),
```
then `dists` would correspond to a list containing `π_1` and `π_2` and `idx`
would be a list containing `1:k` and `(k+1):n`.

    ProductPrior(dists, dims)

Base consructor specifying a list of prior dsitributions `dists` and a
corresponding number of dimensions to which each prior distribution refers to.
"""
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
