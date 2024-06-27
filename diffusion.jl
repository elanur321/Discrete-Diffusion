include("noise_schedule.jl")

struct MaskedDiffusionLanguageModel 

    vocab_size::Int
    mask_token_id::AbstractArray
    α::Function
    model::Function

end

# TODO: change to one-hot vector representations of tokens instead of single numbers

function forward(process::MaskedDiffusionLanguageModel, x_s::AbstractArray, s::Real, t::Real)\
    """
    Function for forward masking process described in chapter 3.2.1
    """
    z_t = copy(x_s)
    for (i, value) in enumerate(x_s)

        p_keep = process.α(t)

        if rand() < p_keep
            z_t[i] = value
        else z_t[i] = process.mask_token_id end
    end
    return z_t
end

" Possible implementation more efficient for GPU computations
using CUDA

function forward(process::MaskedDiffusionLanguageModel, x_s::AbstractArray, s::Real, t::Real)
    p_keep = process.α(t)
    rand_vals = CUDA.rand(eltype(x_s.data), size(x_s.data))
    
    # Create a new mask where rand_vals < p_keep
    new_mask = rand_vals .< p_keep

    # Combine the new mask with the existing mask
    combined_mask = new_mask .| vec(x_s.indices)

    # Create a new MaskedArray with the combined mask
    z_t = MaskedArray(x_s.data, findall(combined_mask))

    # Set all newly masked values to the mask token
    newly_masked = findall(.!new_mask .& .!vec(x_s.indices))
    z_t.data[newly_masked] .= process.mask_token_id
    
    return z_t
end
"

#TODO: check vector solution for optimising GPU by claude(in poe)
" # Find indices of masked tokens  
    masked_indices = findall(x -> x == process.mask_token_id, x_t)
    
    if !isempty(masked_indices)
        # Denoising step for all masked tokens at once
        x_theta = process(x_t, t)
        
        # Calculate probabilities for all tokens except the mask token, for all masked positions
        probs = zeros(vocab_size, length(masked_indices))
        probs[1:vocab_size-1, :] .= (1 - alpha_s) .* process.mask_vector[1:vocab_size-1] .+ 
                    (alpha_s - alpha_t) .* x_theta[1:vocab_size-1, masked_indices]
        probs ./= (1 - alpha_t)
        
        # Zero Masking Probabilities: Set the probability of the mask token to 0
        probs[process.mask_token_id, :] .= 0
        
        # Renormalize
        probs ./= sum(probs, dims=1)
        
        # Sample tokens from the categorical distribution for all masked positions
        x_s[masked_indices] = [rand(Categorical(probs[:, i])) for i in 1:size(probs, 2)]
    end"


function backward(process::MaskedDiffusionLanguageModel, x_t::AbstractArray, s::Real, t::Real)
    """Function for reverse unmasking process described in chapter 3.2.2"""

    @assert 0 ≤ s < t ≤ 1 "Invalid time steps: require 0 ≤ s < t ≤ 1"

    vocab_size = size(process.embedding, 1)
    x_s = copy(x_t)
    
    alpha_s = process.α(s)
    alpha_t = process.α(t)  
    
    for i in 1:length(x_t)      
        if x_t[i] != process.mask_token_id
            # Carry-Over Unmasking: If the token is not masked, keep it unchanged
            x_s[i] = x_t[i]
            
        else
        "Mistrl medium (ai) suggestion"  #TODO: check if better
                    
        "probs = zeros(vocab_size)
        logits = (1 - alpha_s) .* log.(process.mask_vector[1:vocab_size-1]) + 
                (alpha_s - alpha_t) .* x_theta[1:vocab_size-1, i]
        probs[1:vocab_size-1] = softmax(logits)
        # Zero masking probabilities
        probs[process.mask_token_id] = 0
        x_s[i] = rand(Categorical(probs))"

            # Denoising step:
            x_theta = process.model(x_t, t) 
            # Calculate probabilities for all tokens except the mask token
            probs = zeros(vocab_size)
            probs[1:vocab_size-1] = (1 - alpha_s) * process.mask_vector[1:vocab_size-1] + 
                        (alpha_s - alpha_t) * x_theta[1:vocab_size-1, i]
            probs ./= (1 - alpha_t)   

            # Zero Masking Probabilities: Set the probability of the mask token to 0
            probs[process.mask_token_id, :] .= 0
        
            # Renormalize
            probs ./= sum(probs, dims=1)
            
            # Sample a token from the categorical distribution
            x_s[i] = rand(Categorical(probs))
        end
    end

    return x_s
end

_sampleforward(rng::AbstractRNG, process::MaskedDiffusionLanguageModel, t::Real, x::AbstractArray) =
    sample(rng, forward(process, x, 0, t))

function _endpoint_conditioned_sample(rng::AbstractRNG, process::MaskedDiffusionLanguageModel, s::Real, t::Real, x_0::AbstractArray, x_t::AbstractArray)
    prior = forward(process, x_0, 0, s)
    likelihood = backward(process, x_t, s, t)
    return sample(rng, combine(prior, likelihood))
end