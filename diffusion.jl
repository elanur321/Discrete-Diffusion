include("noise_schedule.jl")

struct MaskedDiffusionLanguageModel 

    vocab_size::Int
    mask_token_id::Int
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


############# Example #################

vocab = [1, 2, 3]
ex = MaskedDiffusionLanguageModel(length(vocab), length(vocab) + 1, cosineschedule)

@show vocab = forward(ex, vocab, 0, 0.1)
@show vocab = forward(ex, vocab, 0, 0.2)
@show vocab = forward(ex, vocab, 0, 0.3)

######################################

function backward(process::MaskedDiffusionLanguageModel, x_theta::AbstractArray, alpha_s::Real, alpha_t::Real)
    """Function for reverse unmasking process described in chapter 3.2.2"""
    k = length(x_theta)
    
    z_t = copy(x_t)
    
    for i in eachindex(z_t)         #TODO: figure out how eachindex() works
        if z_t[i] != process.masked_token_ID # SUBS parameterization
            z_s[i] = z_t[i]

        else z_t[i] == process.masked_token_ID   

            probs = zeros(k)
            probs[1:K-1] = (process.α(s) - process.α(t)) * x_theta[1:K-1]
            probs[K] = 1 - process.α(s)
            probs ./= (1 - process.α(t))

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