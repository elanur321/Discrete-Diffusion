
using Flux
using DataFrames
using Random
using StaticArrays
using Distributions

rng = Random.default_rng()

# include("diffusion.jl")
include("loss.jl")
include("Diffusions.jl/src/maskedarrays.jl")
include("Diffusions.jl/src/randomvariable.jl")
include("Diffusions.jl/src/utils.jl")
"""
struct CategoricalVariables{K, T}
    p::Array{SVector{K, T}}
end

ncategories(::CategoricalVariables{K}) where K = K

Base.size(X::CategoricalVariables) = size(X.p)"""

function forward(process::MaskedDiffusionLanguageModel, x::AbstractArray, s::Real, t::Real)
    α_t = process.α(t)
    
    # Create output probabilities
    probs = similar(x, Float32)
    
    for i in eachindex(x)
        if x[i] == process.mask_token_id
            probs[i] = 1.0  # Masked tokens stay masked
        else
            probs[i] = α_t  # Probability of keeping original token
        end
    end
    
    return CategoricalVariables([Categorical([p, 1-p]) for p in probs])
end

_sampleforward(rng::AbstractRNG, process::MaskedDiffusionLanguageModel, t::Real, x::AbstractArray) =
    sample(rng, forward(process, x, 0, t))


sampleforward(process, t, x) = sampleforward(Random.default_rng(), process, t, x)

sampleforward(
    rng::AbstractRNG,
    process,
    t::Union{Real,AbstractVector{<:Real}},
    x::NTuple{N,AbstractArray}
) where {N} = sampleforward.(rng, process, (t,), x)

function sampleforward(rng::AbstractRNG, process, t::Real, x::AbstractArray)
    x = copy(x)
    @show x
    @show maskedvec(x)
    maskedvec(x) .= _sampleforward(rng, process, t, maskedvec(x))
    return x
end

function sampleforward(rng::AbstractRNG, process, t::AbstractVector{<:Real}, x::AbstractArray)
    x = copy(x)
    for i in axes(x, ndims(x))
        maskedvec(x, i) .= _sampleforward(rng, process, t[i], maskedvec(x, i))
    end
    return x
end

###################################################################################################

function forward(process::MaskedDiffusionLanguageModel, x::AbstractMatrix, s::Real, t::Real)
    α_t = process.α(t)[1]
    
    probs = similar(x, Float32)
    for i in 1:size(x, 1)
        if x[i, :] == process.mask_token_id
            probs[i, :] = x[i, :]  # Keep mask token as is
        else
            probs[i, :] = α_t * x[i, :] + (1 - α_t) * process.mask_token_id
        end
    end

    categorical_probs = [SVector{size(x, 2), Float32}(probs[i, :]) for i in 1:size(probs, 1)]
    
    return CategoricalVariables(categorical_probs)
end

vocab_size = 2
mask_token = [0.0, 0.0, 1.0]
process = MaskedDiffusionLanguageModel(vocab_size, mask_token, linear)

x = Float32[
    1 0 0;  # Token 1
    0 1 0;  # Token 2
    0 0 1;  # Masked token
    1 0 0;  # Token 1
    0 1 0   # Token 2
]

x̂ = Float32[
    0 1 0;  # Token 1
    0 1 0;  # Token 2
    1 0 0;  # Masked token
    1 0 0;  # Token 1
    0 1 0   # Token 2
]

# @show _sampleforward(Random.default_rng(), process, 0.5, x)

@show standardloss(process, 0.5, x̂, x)