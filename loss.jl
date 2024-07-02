# Implementation of the Masked Diffusion Language Model loss function
using Zygote
using Random
using Flux.Losses
using LinearAlgebra
using Test
using Statistics

include("noise_schedule.jl")
include("diffusion.jl")


# TODO: examining only the masked token indices rather than comparing the full true and approximate posterior distributions.

########################### TEST ################################

defaultscaler(x) = 1

# This function is an attempt at a loss function allowing for batching.
# However, this is unecessarily complicated and a much smoother approach
# is simply to iterate over a number of time steps and call the loss 
# function many times as is shown in evaluation.ipynb
function standardloss(
    p::MaskedDiffusionLanguageModel,
    t::Union{Real,AbstractVector{<:Real}},
    x̂, x;
    scaler=defaultscaler
)
    # Ensure x̂ and x have the same size
    @assert size(x̂) == size(x) "Dimensions of x̂ and x must match"

    # @show p.α(t)
    α_t, α_prime = p.α(t)
    scaling_factor = α_prime ./ (1 .- α_t)

    if ndims(x̂) == 2  # Single batch case
        # @show "SINGLE BATCH"

        @assert length(scaling_factor) == 1 "For single batch, scaling_factor should be a single value"

        # Compute logitcrossentropy for the single batch
        loss = logitcrossentropy(x̂, x)

        # Apply scaling factor before summing
        scaled_loss = scaling_factor[1] .* loss

        return [sum(scaled_loss)]  # Return as a single-element vector for consistency
    else  # Multiple batches case
         # @show "BIG BATCH"
        # @assert size(x̂, 3) == length(scaling_factor) "Number of batches must match length of scaling_factor"

        # Compute logitcrossentropy for all batches at once
        losses = logitcrossentropy.(eachslice(x̂, dims=3), eachslice(x, dims=3))

        # Apply scaling factor to each batch before summing
        # Reshape scaling_factor to broadcast correctly
        # scaled_losses = losses .* reshape(scaling_factor, (1, 1, :))
        scaled_losses = losses .* scaling_factor

        # Sum the scaled losses for each batch
        batch_losses = vec(sum(scaled_losses, dims=(1,2)))

        return batch_losses
    end
end

# By following what we said above, this function easily computes the loss for a single
# time point t given two arrays {x, x̂} representing the true distribution and the predicted
# values, respectively.
function standardloss1(
    p::MaskedDiffusionLanguageModel,
    t::Union{Real,AbstractVector{<:Real}},
    x̂, x;
    scaler=defaultscaler
)
    α_t, α_gradient = p.α(t)
    scale = (α_gradient)/(1-α_t)
    return scale * logitcrossentropy(x̂, x)
end

p = MaskedDiffusionLanguageModel(5, 5, linear)


@show standardloss1(p, 0.5, [1, 2, 3], [4, 5, 6])

@show standardloss(p, 0.5, [1, 2, 3], [4, 5, 6])