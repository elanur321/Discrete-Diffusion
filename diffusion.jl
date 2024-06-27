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
function forward(process::MaskedDiffusionLanguageModel, x_s::CuArray, t::Real)
    p_keep = process.α(t)
    mask = CUDA.rand(Float32, size(x_s)) .< p_keep
    z_t = CUDA.ifelse.(mask, x_s, process.mask_token_id)
    return z_t
end
"

function backward(process::MaskedDiffusionLanguageModel, x_t::AbstractArray, s::Real, t::Real)
    """Function for reverse unmasking process described in chapter 3.2.2"""
    vocab_size = size(process.embedding, 1)
    x_s = copy(x_t)
    
    alpha_s = compute_alpha(process, s)
    alpha_t = compute_alpha(process, t)
    
   
    
    for i in eachindex(x_t)      #TODO: figure out how eachindex() works
        if x_t[i] != process.mask_token_id
            x_s[i] = x_t[i]
            
        else
            x_theta = process(x_t, t) 

            
            
            probs = zeros(vocab_size)
            probs[1:vocab_size-1] = (1 - alpha_s) * process.mask_vector[1:vocab_size-1] + 
                        (alpha_s - alpha_t) * x_theta[1:vocab_size-1, i]
            probs ./= (1 - alpha_t)
            # Zero masking probabilities
            probs[process.mask_token_id] = 0
            
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
