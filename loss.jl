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

    if size(x̂[1]) == ()
        loss = logitcrossentropy(x̂, x)
        scaled_loss = scaling_factor * loss
    else
        batch_loss = logitcrossentropy.(x̂, x)
        scaled_loss = scaling_factor .* batch_loss
    end

    return scaled_loss
end

e = MaskedDiffusionLanguageModel(1, [0, 0, 1], linear, defaultscaler)

@show standardloss(e, [0.2, 0.2], [[0.4,0.6], [0.2, 0.8]], [[1,0],[0,1]])

@show standardloss(e, 0.2, [0.2, 0.8], [0, 1])

