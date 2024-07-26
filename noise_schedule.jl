# Define the noise schedule functions 

function cosineschedule(t::Union{Real,AbstractVector{<:Real}})
    return cos.(pi .* t ./ 2), (pi/2).*(-sin.(pi .* t ./ 2))
end

function loglinear(t::Union{Real,AbstractVector{<:Real}})
    return exp.(log10.(1 .- t)), exp.(1 .- t).*(1 ./ (log(10)t .- log(10)))
end

function linear(t::Union{Real,AbstractVector{<:Real}})
    return 1 .- t, - t ./ t
end

function loglinear(t::Union{Real,AbstractVector{<:Real}}; eps::Float64=1e-3)
    σ = -log1p.(-(1 - eps) .* t)
    dσ_dt =(1 - eps) ./ (1 .- (1 - eps) .* t)
    return exp.(-σ), -dσ_dt.*exp.(-σ)
end
