include("noise_schedule.jl")

struct MaskedDiffusionLanguageModel 

    vocab_size::Int
    mask_token_id::Int
    noise_schedule::Function

end

function forward(process::MaskedDiffusionLanguageModel, x_s::AbstractArray, t::Real)\
    """
    Function for forward masking process described in chapter 3.2.1
    """
    z_t = similar(x_s)
    for (i, value) in enumerate(x_s)

        p_keep = α_t(t, process.noise_schedule)

        if rand() < p_keep
            z_t[i] = value
        else z_t[i] = process.mask_token_id end
    end
    return z_t
end


############# Example #################

vocab = [1, 2, 3]
ex = MaskedDiffusionLanguageModel(length(vocab), length(vocab) + 1, loglinear)

@show vocab = forward(ex, vocab, 0.1)
@show vocab = forward(ex, vocab, 0.2)
@show vocab = forward(ex, vocab, 0.3)

######################################

function backward(process::MaskedDiffusionLanguageModel, x_t::AbstractArray, s::Real, t::Real)
    """Function for reverse unmasking process described in chapter 3.2.2"""
    k =
    z_t = copy(x_t)
    
    for i in eachindex(z_t)
    if z_t == masked

    end

    return z_t
end
