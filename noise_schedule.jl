# Define the noise schedule functions 

function loglinear(t::Real)
    return -log10(1-t)
end

function cosinesquared(t::Real)
    return -log10((cos((pi/2)*(1-t)))^2)
end

function cosineschedule(t::Real)
    return -log10(cos((pi/2)*(1-t)))
end

function linear(t::Real, σ_max = 10^8)
    return σ_max * t
end


function α_t(t::Real, σ::Function = linear)

    """Calculates the noise schedule parameter α_t.


        # Arguments
        - `t::Real`: The time step, a real number in the range [0, 1].
        - `σ::Function`: A function representing the noise schedule. 
        Defaults to `linear`.

        # Returns
        - A real number representing the noise schedule parameter α_t at time `t`.

        # Noise schedule options
        - `linear(t)`: Linear noise schedule.
        - `cosineschedule(t)`: Cosine noise schedule.
        - `cosinesquared(t)`: Cosine squared noise schedule.
        - `loglinear(t)`: Log-linear noise schedule.
    """

    return exp(-σ(t))
end
