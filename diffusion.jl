include("noise_schedule.jl")

struct MaskedDiffusionLanguageModel 

    vocab_size::Int
    mask_token_id::AbstractArray
    α::Function

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

_sampleforward(rng::AbstractRNG, process::MaskedDiffusionLanguageModel, t::Real, x::AbstractArray) =
    sample(rng, forward(process, x, 0, t))


function _endpoint_conditioned_sample(rng::AbstractRNG, process::MaskedDiffusionLanguageModel, s::Real, t::Real, x_0::AbstractArray, x_t::AbstractArray)
    @assert 0 ≤ s < t ≤ 1 "Invalid time steps: require 0 ≤ s < t ≤ 1" #not sure if this is needed but il keep it here
    
    prior = forward(process, x_0, 0, s)

    # Move data to GPU
    x_0 = CuArray(x_0), x_t = CuArray(x_t)

    vocab_size = size(process.embedding, 1) #TODO: check this
    x_s = copy(x_t)
    
    alpha_s = process.α(s) # TODO: This function has been changed, now returns (value of noise noise_schedule, gradient of noise_schedule)
    alpha_t = process.α(t)  

    # Create a mask for non-masked tokens
    non_masked = x_t .!= process.mask_token_id

    # Process all tokens at once
    x_theta = process(x_t, t) # TODO: I think you don't need to predict using any type of process here, but you can instead just use x_0 as passed to the function

    # Compute unnormalized log probabilities for non-masked tokens
    # Compute logits for all tokens
    logits = (1 - alpha_s) .* log.(process.mask_vector[1:vocab_size-1]) .+ 
             (alpha_s - alpha_t) .* x_theta[1:vocab_size-1, :]

    # Normalize using softmax
    # Compute probabilities using softmax
    probs = vcat(softmax(logits, dims=1), zeros(1, size(logits, 2)))    #TODO: understand softmax better

    # Zero out probabilities for mask token
    probs[process.mask_token_id, :] .= 0

    # Sample tokens from categorical distribution
    sampled_tokens = [rand(Categorical(probs[:, i])) for i in eachindex(probs, 2)]  #TODO: learn what exactly rand(categorical(probs)) does when choosing

    # Combine non-masked tokens and sampled tokens
    x_s = ifelse.(non_masked, x_t, sampled_tokens)


    #old non vectorised code. keeping untill sure the optimizations works (no guarantee the old code works either)
    "
    for i in eachindex(x_t)      
        if x_t[i] != process.mask_token_id
            # Carry-Over Unmasking: If the token is not masked, keep it unchanged
            x_s[i] = x_t[i]
        else

            x_theta = process(x_t, t) 

            # Compute unnormalized log probabilities for non-masked tokens
            logits = (1 - alpha_s) .* log.(process.mask_vector[1:vocab_size-1]) .+ 
                     (alpha_s - alpha_t) .* x_theta[1:vocab_size-1, i]
            
            # Normalize using softmax
            probs = zeros(vocab_size)
            probs[1:vocab_size-1] = softmax(logits) #TODO: understand softmax better
            
            # Zero masking probabilities
            CUDA.@inbounds probs[process.mask_token_id, :] .= 0
            
            # Sample a token from the categorical distribution
            x_s[i] = rand(Categorical(probs))  #TODO: learn wahate exactly rabd(categorical(probs)) does when choosing
        end
    end
    "
    "return x_s" #from backard

    return sample(rng, combine(prior, x_s)) #quick merge

    "return sample(rng, combine(prior, likelihood))" #prev
end