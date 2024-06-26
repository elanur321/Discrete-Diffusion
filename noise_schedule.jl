# Define the noise schedule functions 


function α_t(t::Real)

    """Calculates the noise schedule parameter α_t based on the cosine noise scheduler

        # Arguments
        - `t::Real`: The time step, a real number in the range [0, 1].

        # Returns
        - A real number representing the noise schedule parameter α_t at time `t`.
    """

    # return exp(log(1-t)) <--- Implementation of loglinear noise scheduler

    return cos(pi * t/2)
end
