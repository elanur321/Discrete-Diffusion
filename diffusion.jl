
struct MaskedDiffusionLanguageModel

end

function forward(process::MaskedDiffusionLanguageModel, x_s::AbstractArray, s::Real, t::Real)\
    """
    Function for forward masking process described in chapter 3.2.1
    """
end

function backward(process::MaskedDiffusionLanguageModel, x_t::AbstractArray, s::Real, t::Real)
    """
    Function for reverse unmasking process described in chapter 3.2.2
    """
end
