# Define the noise schedule functions 

function cosineschedule(t::Union{Real,AbstractVector{<:Real}})
    return cos(pi * t/2)
end

function loglinear(t::Union{Real,AbstractVector{<:Real}})
    return exp.(log10.(1 .- t))
end

function linear(t::Union{Real,AbstractVector{<:Real}})
    return 1 .- t
end
