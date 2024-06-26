# Define the noise schedule functions 

function α_t(t::Real, σ::Function = linear)

    """Calculates the noise schedule parameter α_t based on the cosine noise scheduler

        # Arguments
        - `t::Real`: The time step, a real number in the range [0, 1].
        - `σ::Function`: A function representing the noise schedule. 
        Defaults to `linear`.

        # Returns
        - A real number representing the noise schedule parameter α_t at time `t`.
    """

    return cos(pi * t/2)
end
