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


function _sampleforward(rng::AbstractRNG, process::MaskedDiffusionLanguageModel, t::Real, x::AbstractArray)

    z_t = copy(x)
    for (i, value) in enumerate(x)

        p_keep = process.α(t)[1]

        if rand() < p_keep
            z_t[i] = value
        else z_t[i] = process.mask_token_id end
    end
    return z_t

end
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
end


# This implementation works for data that is NOT one-hot-encoded.
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


################################### One-hot-encoded ################################################

# According to the article, these functions should work on one-hot-encoded data. 
# However, the above functions don't. 
# Bellow are modified functions which should however serve this purpose.

# TODO: potentially vectorize operations and optimize for GPU computations 

function _sampleforward(rng::AbstractRNG, process::MaskedDiffusionLanguageModel, t::Real, x::AbstractVector{Bool})
    # Get the keep probability
    α_t = process.α(t)[1]  # Extract the first element of the tuple
    p_keep = α_t  # Assuming α_t directly gives the keep probability
    
    vocab_size = process.vocab_size  # Assume this is a field in your MaskedDiffusionLanguageModel
    num_tokens = length(x) ÷ vocab_size
    
    result = falses(length(x))
    
    for i in 1:num_tokens
        token_slice = (i-1)*vocab_size + 1 : i*vocab_size
        if rand(rng) < p_keep
            # Keep the original token
            result[token_slice] = x[token_slice]
        else
            # Mask the token
            result[token_slice[process.mask_token_id]] = true
        end
    end
    
    return result
end

function _sampleforward(rng::AbstractRNG, process::MaskedDiffusionLanguageModel, t::Real, x::AbstractArray{Bool, 3})
    vocab_size, batch_size, seq_length = size(x)
    
    # Get the keep probability
    α_t = process.α(t)[1]  # Extract the first element of the tuple
    p_keep = α_t  # Assuming α_t directly gives the keep probability
    
    result = falses(size(x))
    
    for k in 1:seq_length
        for j in 1:batch_size
            if rand(rng) < p_keep
                # Keep the original token
                result[:, j, k] = x[:, j, k]
            else
                # Mask the token
                result[process.mask_token_id, j, k] = true
            end
        end
    end
    
    return result
end

function _endpoint_conditioned_sample(
    rng::AbstractRNG, 
    process::MaskedDiffusionLanguageModel, 
    s::Real, 
    t::Real, 
    x0::AbstractVector{Bool}, 
    xt::AbstractVector{Bool}
)
    @assert 0 ≤ s < t ≤ 1 "Invalid time steps: require 0 ≤ s < t ≤ 1"
    @assert length(x0) == length(xt) "x0 and xt must have the same length"
    
    vocab_size = process.vocab_size
    sequence_length = length(xt) ÷ vocab_size
    xs = copy(xt)
    
    α_s = process.α(s)[1]
    α_t = process.α(t)[1]

    for i in 1:sequence_length
        token_slice = (i-1)*vocab_size + 1 : i*vocab_size
        if xt[token_slice[process.mask_token_id]]
            probs = zeros(Float32, vocab_size)
            probs[process.mask_token_id] = 1 - α_s # CHECK
            predicted_token = findfirst(x0[token_slice])
            probs[predicted_token] = α_s - α_t
            probs ./= sum(probs)
            sampled_token = rand(rng, Categorical(probs))
            xs[token_slice] .= false
            xs[token_slice[sampled_token]] = true
        end
    end

    return xs
end