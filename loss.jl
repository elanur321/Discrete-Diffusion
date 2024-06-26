# Implementation of the Masked Diffusion Language Model loss function

function standardloss(
    p::MaskedDiffusionLanguageModel,
    t::Union{Real, AbstractVector{<: Real}},
    x̂, x;
    scaler = defaultscaler
)

    return 
end