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

defaultscaler(x) = 1

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
        # Compute logitcrossentropy for the single batch
        loss = logitcrossentropy(x̂, x)
        scaled_loss = scaling_factor[1] .* loss

        return [mean(scaled_loss)]  # Return as a single-element vector for consistency
    else  # Multiple batches case
        # Compute logitcrossentropy for all batches at once
        losses = logitcrossentropy.(eachslice(x̂, dims=3), eachslice(x, dims=3))
        scaled_losses = losses .* scaling_factor

        # Take the mean of the scaled losses for each batch
        batch_losses = vec(mean(scaled_losses, dims=(1,2))) # alternatively vec(sum(scaled_losses, dims=(1,2)))
        return batch_losses
    end
end
