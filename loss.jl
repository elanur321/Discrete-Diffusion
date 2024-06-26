# Implementation of the Masked Diffusion Language Model loss function
using Zygote
using CUDA
using LinearAlgebra
include("noise_schedule.jl")
include("diffusion.jl")

# TODO: Actually run code on Nvidia GPU to see if it works

function standardloss(
    p::MaskedDiffusionLanguageModel,
    t::Union{Real, AbstractVector{<: Real}},
    x̂, x;
)
    return (gradient(x -> α_t(x), t))[1]/(1 - α_t(t)) * sum([log(x̂[n] ⋅ value) for (i, value) in enumerate(x)])
end

############# Example #################

# ex = MaskedDiffusionLanguageModel(3, 4)

# standardloss(ex, 0.2, [2, 3, 1], [1, 2, 3])

######################################

function optimized_loss(
    p::MaskedDiffusionLanguageModel,
    x̂::CuMatrix, 
    x::CuMatrix;
    num_samples::Int = 100
)
    # Sample t values uniformly between 0 and 1
    t_samples = CUDA.rand(Float32, num_samples)
    
    # Compute α_t and its gradient for all sampled t values
    α_t_vals = α_t.(t_samples)
    grad_α_t = gradient(α_t, t_samples)[1]

    dot_products = sum(x̂ .* x, dims=1)
    log_probs = CUDA.log.(dot_products)
    total_log_prob = sum(log_probs)
    
    # Compute the loss for each t sample
    sample_losses = (grad_α_t ./ (1 .- α_t_vals)) .* total_log_prob
    
    # Average the losses
    loss = mean(sample_losses)
    
    return loss
end

@show optimized_loss(ex, 0.2, CuArray([2, 3, 1]), CuArray([1, 2, 3]))