# Implementation of the Masked Diffusion Language Model loss function
using Zygote
using Random
using Flux.Losses
using CUDA
using LinearAlgebra
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
    return (α_prime/(1 .- α_t)) .* logitcrossentropy(x̂, x; dims = 1, agg = mean)
end
