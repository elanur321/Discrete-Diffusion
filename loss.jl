# Implementation of the Masked Diffusion Language Model loss function
using Zygote
using CUDA
using LinearAlgebra
include("noise_schedule.jl")
include("diffusion.jl")

# TODO: Actually run code on Nvidia GPU to see if it works
# TODO: Not sure if x̂ and x are CuMatrices och just normal Matrices

function defaultscaler(p::MaskedDiffusionLanguageModel, t::Union{Real,AbstractVector{<:Real}})
    α_t = p.α_t(t)
    α_prime = gradient(p.α_t, t)[1]
    return α_prime / (1 - α_t)
end

function standardloss(
    p::MaskedDiffusionLanguageModel,
    t::Union{Real,AbstractVector{<:Real}},
    x̂::CuMatrix, 
    x::CuMatrix;
    scaler=defaultscaler
)

    α_t = p.α(t)
    α_prime = gradient(p.α, t)[1]
    
    return( α_prime / (1 - α_t)) * log.(sum(x̂ .* x, dims=1))
end
