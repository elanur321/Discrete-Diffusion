include("noise_schedule.jl")

struct MaskedDiffusionLanguageModel 

    vocab_size::Int
    mask_token_id
    α::Function

end

# TODO: change to one-hot vector representations of tokens instead of single numbers

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
    
    prior = forward(process, x_0, 0, s)

    # Move data to GPU
    x_0 = CuArray(x_0), x_t = CuArray(x_t)

    vocab_size = size(process.embedding, 1)
    x_s = copy(x_t)
    
    alpha_s = process.α(s)[1] # TODO: This function has been changed, now returns (value of noise noise_schedule, gradient of noise_schedule)
    alpha_t = process.α(t)[1]  

    # Create a mask for non-masked tokens
    non_masked = x_t .!= process.mask_token_id

    "x_theta = process(x_t, t) "# TODO: I think you don't need to predict using any type of process here, but you can instead just use x_0 as passed to the function
    #ah thats a remnant of old code i believe, il keep it in quotationmarks for now just in case

    # Compute unnormalized log probabilities for all non-masked tokens
    logits = (1 - alpha_s) .* log.(process.mask_vector[1:vocab_size-1]) .+ (alpha_s - alpha_t) .* x_0[1:vocab_size-1, :]

    # Normalize / Compute probabilities using softmax
    probs = vcat(softmax(logits, dims=1), zeros(1, size(logits, 2)))   

    # Zero out probabilities for mask token
    probs[process.mask_token_id, :] .= 0

    # Sample tokens from categorical distribution
    sampled_tokens = [rand(Categorical(probs[:, i])) for i in eachindex(probs, 2)]  #TODO: learn what exactly rand(categorical(probs)) does when choosing

    # Combine non-masked tokens and sampled tokens
    x_s = ifelse.(non_masked, x_t, sampled_tokens)


   
    #old non vectorised code. keeping untill sure the optimizations works (no guarantee the old code works either)
    "for i in eachindex(x_t)      
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
    end"

    return sample(rng, combine(prior, x_s)) #quick merge


end