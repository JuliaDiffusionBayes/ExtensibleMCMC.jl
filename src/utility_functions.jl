#===============================================================================
        Additional utility functions, not tied to the package
===============================================================================#

# these appear in multiple places of this suite, TODO define them only once
ismutable(el) = ismutable(typeof(el))
ismutable(::Type) = Val(false)
ismutable(::Type{<:Array}) = Val(true)

"""
    _assure_scalar(v)

Accept a scalar or a vector of length one and return it as a scalar. Raise
assertion error if other objects are passed.
"""
function _assure_scalar(v)
    typeof(v) <: Number && return v
    @assert length(v) == 1
    @assert typeof(v[1]) <: Number
    first(v)
end

"""
    _upgrade_to_vec(v, N)

Receive a scalar or a vector of length 1 and return a length `N` vector of
repeats of entry v.
"""
function _upgrade_to_vec(v, N)
    typeof(v) <: Vector && length(v) == N && return v
    @assert length(v) == 1
    repeat([first(v)], N)
end

"""
    _upgrade_to_svec(v, ::Val{N}) where N

Receive a scalar or a vector of length 1 and return a length `N` static vector
of repeats of entry v.
"""
function _upgrade_to_svec(v, ::Val{N}) where N
    typeof(v) <: SVector && length(v) == N && return v
    length(v) == N && return SVector{N}(v)
    @assert length(v) == 1
    first(v) .+ zero(SVector{N})
end

"""
    sigmoid(x, a=1.0)

Sigmoid function. (Inverse of logit).
"""
sigmoid(x, a=1.0) = 1.0 / (1.0 + exp(-a*x))

"""
    logit(x, a=1.0)

Logit function. (Inverse of sigmoid).
"""
logit(x, a=1.0) = (log(x) - log(1-x))/a

"""
    custom_zero(x::T, ::Type{elT}) where {T,elT}

Create a `zero` with eltype `elT`, that is of the same structure as the
collection `x`.

# Examples
```julia-repl
julia> custom_zero(3.0, Bool)
false
julia> custom_zero([3.0, 4.0], Bool)
2-element Array{Bool,1}:
 0
 0
julia> custom_zero(SVector{2}(1, 2), ComplexF64)
2-element SArray{Tuple{2},Complex{Float64},1,2} with indices SOneTo(2):
 0.0 + 0.0im
 0.0 + 0.0im
```
"""
function custom_zero(x::T, ::Type{elT}) where {T,elT}
    T <: Array && return zeros(elT, size(x))
    T <: Number && return zero(elT)
    T <: SArray && return zero(similar_type(T, elT))
end
