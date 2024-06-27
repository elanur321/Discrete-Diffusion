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
    x_0 = model(z_t, t)  # This is our guess at the original text

    vocab_size

  
    
    z_t = copy(x_0)
    
    for i in eachindex(z_t)         #TODO: figure out how eachindex() works
        if z_t[i] != model.mask_token_id
            z_s[i] = z_t[i]
            
        else z_t[i] == process.mask_token_id   

            x_theta = model(z_t, t)  # This should be your neural network
            
            probs = zeros(vocab_size)
            probs[1:vocab_size-1] = (1 - alpha_s) * model.mask_vector[1:vocab_size-1] + 
                                    (alpha_s - alpha_t) * x_theta[1:vocab_size-1, i]
            probs ./= (1 - alpha_t)
            
            # Zero masking probabilities
            probs[model.mask_token_id] = 0
            
            z_s[i] = rand(Categorical(probs))
        end
    end

    return z_s
end

_sampleforward(rng::AbstractRNG, process::MaskedDiffusionLanguageModel, t::Real, x::AbstractArray) =
    sample(rng, forward(process, x, 0, t))

function _endpoint_conditioned_sample(rng::AbstractRNG, process::MaskedDiffusionLanguageModel, s::Real, t::Real, x_0::AbstractArray, x_t::AbstractArray)
    prior = forward(process, x_0, 0, s)
    likelihood = backward(process, x_t, s, t)
    return sample(rng, combine(prior, likelihood))
end
