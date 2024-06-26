# Implementation of the Masked Diffusion Language Model loss function
using Zygote
using LinearAlgebra
include("noise_schedule.jl")
include("diffusion.jl")

# TODO: optimize for GPU & take average over several time steps - i.e. implement integration part of equation

function standardloss(
    p::MaskedDiffusionLanguageModel,
    t::Union{Real, AbstractVector{<: Real}},
    x̂, x;
)
    return (gradient(x -> α_t(x), t))[1]/(1 - α_t(t)) * sum([log(x̂[n] ⋅ value) for (i, value) in enumerate(x)])
end

############# Example #################

ex = MaskedDiffusionLanguageModel(3, 4)

standardloss(ex, 0.2, [2, 3, 1], [1, 2, 3])

######################################