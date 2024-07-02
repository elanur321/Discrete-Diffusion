include("noise_schedule.jl")
using Random
using StatsBase
using Distributions

abstract type Process end

Base.broadcastable(x::Process) = Ref(x)

abstract type TractableProcess <: Process end #Tractable uncertainty propogation
abstract type SamplingProcess <: Process end #Only deals with point masses and sampling

abstract type GaussianStateProcess <: TractableProcess end
abstract type DiscreteStateProcess <: TractableProcess end

struct MaskedDiffusionLanguageModel <: DiscreteStateProcess

    vocab_size::Int
    mask_token_id
    α::Function

end

# TODO: change to one-hot vector representations of tokens instead of single numbers
"""
function _sampleforward(rng::AbstractRNG, process::MaskedDiffusionLanguageModel, t::Real, x::AbstractArray)

    z_t = copy(x)
    for (i, value) in enumerate(x)

        p_keep = process.α(t)[1]

        if rand() < p_keep
            z_t[i] = value
        else z_t[i] = process.mask_token_id end
    end
    return z_t

end"""

" Possible implementation more efficient for GPU computations
using CUDA

function _sampleforward(rng::AbstractRNG, process::MaskedDiffusionLanguageModel, t::Real, x::AbstractArray)
    # Move data to GPU
    z_t = CUDA.fill(process.mask_token_id, size(x))
    x_d = CuArray(x)
    p_keep = process.α(t)[1]

    # Generate random numbers on the GPU
    rand_vals = CUDA.rand(size(x_d))

    # Apply mask condition
    mask = rand_vals .< p_keep
    z_t[mask] .= x_d[mask]

    return Array(z_t)  # Move data back to CPU if needed
end
"

"""
function _sampleforward(rng::AbstractRNG, process::MaskedDiffusionLanguageModel, t::Real, x::AbstractArray)
    # Create a copy of x
    z_t = fill(process.mask_token_id, size(x))
    p_keep = process.α(t)[1]

    # Generate random numbers and create a mask
    rand_vals = rand(rng, size(x))
    mask = rand_vals .< p_keep

    # Apply mask condition
    z_t[mask] .= x[mask]

    return z_t
end"""

#_sampleforward(rng::AbstractRNG, process::MaskedDiffusionLanguageModel, t::Real, x::AbstractArray) =
    #sample(rng, forward(process, x, 0, t))

    function _endpoint_conditioned_sample(rng::AbstractRNG, process::MaskedDiffusionLanguageModel, s::Real, t::Real, x_0::AbstractArray, x_t::AbstractArray)
    @assert 0 ≤ s < t ≤ 1 "Invalid time steps: require 0 ≤ s < t ≤ 1" #not sure if this is needed but il keep it here

    # Move data to GPU
    x_0 = CuArray(x_0), x_t = CuArray(x_t)

    vocab_size = size(process.embedding, 1)
    x_s = copy(x_t)
    
    alpha_s = process.α(s)[1]
    alpha_t = process.α(t)[1]  

    # Create a mask for non-masked tokens
    non_masked = x_t .!= process.mask_token_id

    # Compute unnormalized log probabilities for all non-masked tokens
    logits = (1 - alpha_s) .* log.(process.mask_vector[1:vocab_size-1]) .+ (alpha_s - alpha_t) .* x_0[1:vocab_size-1, :]

    # Normalize / Compute probabilities using softmax
    probs = vcat(softmax(logits, dims=1), zeros(1, size(logits, 2)))   

    # make probabilities for mask 0
    probs[process.mask_token_id, :] .= 0

    # Sample tokens from categorical distribution. takes random number 0-1 and choses the word with the probability that matches
    sampled_tokens = [rand(Categorical(probs[:, i])) for i in eachindex(probs, 2)]

    # Combine non-masked tokens and sampled tokens
    x_s = ifelse.(non_masked, x_t, sampled_tokens) #if nonmaksed x_s = x_t else = sampled token
   
    return x_s

    "return sample(rng, combine(prior, x_s))"
end

# Quick-fix implementation
function _endpoint_conditioned_sample(
    rng::AbstractRNG, 
    process::MaskedDiffusionLanguageModel, 
    s::Real, 
    t::Real, 
    x0::AbstractVector{Int}, 
    xt::AbstractVector{Int}
)
    @assert 0 ≤ s < t ≤ 1 "Invalid time steps: require 0 ≤ s < t ≤ 1"
    @assert length(x0) == length(xt) "x0 and xt must have the same length"
    
    vocab_size = process.vocab_size
    sequence_length = length(xt)
    xs = copy(xt)
    
    α_s = process.α(s)[1]
    α_t = process.α(t)[1]

    for i in 1:sequence_length
        if xt[i] == process.mask_token_id
            # For masked tokens, compute probabilities
            probs = zeros(Float32, vocab_size)
            
            # Probability of keeping the mask
            probs[process.mask_token_id] = 1 - α_s
            
            # Probability of sampling the predicted token
            probs[x0[i]] = α_s - α_t
            
            # Normalize probabilities
            probs ./= sum(probs)
            
            # Sample new token
            xs[i] = rand(rng, Categorical(probs))
        end
        # For unmasked tokens, xs[i] remains unchanged (equal to xt[i])
    end

    return xs
end