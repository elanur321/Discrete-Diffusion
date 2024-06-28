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

    losses = map(zip(x̂, x)) do (x̂_batch, x_batch)
        sum(logitcrossentropy.(x̂_batch, x_batch))
    end

    return losses .* scaling_factor
end

### EXAMPLE ###

e = MaskedDiffusionLanguageModel(1, [0, 0, 0, 1], linear)

# @show standardloss(e, 0.2, [[0.4,0.6], [0.2, 0.8]], [[1,0],[0,1]])

# @show standardloss(e, 0.2, [0.2, 0.8], [0, 1])

x = [
    # First sample in batch
    [
        [1, 0, 0, 0],  # First token is category 1
        [0, 1, 0, 0],  # Second token is category 2
        [0, 0, 0, 1]   # Third token is category 4
    ],
    
    # Second sample in batch
    [
        [0, 1, 0, 0],  # First token is category 2
        [0, 0, 1, 0],  # Second token is category 3
        [1, 0, 0, 0]   # Third token is category 1
    ]
]

x̂ = [
    # First sample in batch
    [
        [ 2.0, -1.0,  0.5,  0.1],  # Logits for first token
        [-0.5,  1.5,  0.0, -1.0],  # Logits for second token
        [ 0.2,  0.3, -0.5,  1.0]   # Logits for third token
    ],
    
    # Second sample in batch
    [
        [ 0.5,  1.0, -1.0,  0.2],  # Logits for first token
        [-0.2,  0.1,  2.0, -0.5],  # Logits for second token
        [ 1.5, -1.0,  0.0,  0.5]   # Logits for third token
    ]
]

# standardloss(e, 0.2, x̂, x)

@show sum(logitcrossentropy.(x̂[1], x[1]))

@show losses = map(zip(x̂, x)) do (x̂_batch, x_batch)
    sum(logitcrossentropy.(x̂_batch, x_batch))
end

@show standardloss(e, 0.2, x̂, x)