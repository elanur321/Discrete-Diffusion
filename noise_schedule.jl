# Define the noise schedule functions 

function cosineschedule(t::Real)
    return cos(pi * t/2)
end

function loglinear(t::Real)
    return exp(log(1-t))
end

function linear(t::Real)
    return 1 - t
end