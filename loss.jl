# Implementation of the Masked Diffusion Language Model loss function
using Zygote
using Random
using Flux.Losses
using CUDA
using Flux
using LinearAlgebra
using Test
using Statistics

include("noise_schedule.jl")
include("diffusion.jl")


# TODO: Actually run code on Nvidia GPU to see if it works
# TODO: Not sure if x̂ and x are CuMatrices och just normal Matrices
# TODO: examining only the masked token indices rather than comparing the full true and approximate posterior distributions.

defaultscaler(x) = 1

function standardloss(
    p::MaskedDiffusionLanguageModel,
    t::Union{Real,AbstractVector{<:Real}},
    x̂, x;
    scaler=defaultscaler
)
    α_t, α_prime = p.α(t)

    scaling_factor = α_prime ./ (1 .- α_t)

    if true
        losses = map(zip(x̂, x)) do (x̂_batch, x_batch)
            sum(logitcrossentropy.(x̂_batch, x_batch))
        end
    else

    end

    return losses .* scaling_factor
end

########################### TEST ################################

function standardloss(
    p::MaskedDiffusionLanguageModel,
    t::Union{Real,AbstractVector{<:Real}},
    x̂, x;
    scaler=defaultscaler
)
    # Ensure x̂ and x have the same size
    @assert size(x̂) == size(x) "Dimensions of x̂ and x must match"

    @show p.α(t)
    α_t, α_prime = p.α(t)
    scaling_factor = α_prime ./ (1 .- α_t)

    if ndims(x̂) == 2  # Single batch case
        @show "SINGLE BATCH"

        @assert length(scaling_factor) == 1 "For single batch, scaling_factor should be a single value"

        # Compute logitcrossentropy for the single batch
        loss = logitcrossentropy(x̂, x)

        # Apply scaling factor before summing
        scaled_loss = scaling_factor[1] .* loss

        return [sum(scaled_loss)]  # Return as a single-element vector for consistency
    else  # Multiple batches case
         @show "BIG BATCH"
        @assert size(x̂, 3) == length(scaling_factor) "Number of batches must match length of scaling_factor"

        # Compute logitcrossentropy for all batches at once
        losses = logitcrossentropy.(eachslice(x̂, dims=3), eachslice(x, dims=3))

        # Apply scaling factor to each batch before summing
        # Reshape scaling_factor to broadcast correctly
        scaled_losses = losses .* reshape(scaling_factor, (1, 1, :))

        # Sum the scaled losses for each batch
        batch_losses = vec(sum(scaled_losses, dims=(1,2)))

        return batch_losses
    end
end

