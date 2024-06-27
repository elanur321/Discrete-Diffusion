# Define the noise schedule functions 


function cosineschedule(t::Real)

    """Calculates the noise schedule parameter α_t based on the cosine noise scheduler

        # Arguments
        - `t::Real`: The time step, a real number in the range [0, 1].

        # Returns
        - A real number representing the noise schedule parameter α_t at time `t`.
    """

    return cos(pi * t/2)
end

function loglinear(t::Real)

    """Calculates the noise schedule parameter α_t based on the loglinear noise scheduler

        # Arguments
        - `t::Real`: The time step, a real number in the range [0, 1].

        # Returns
        - A real number representing the noise schedule parameter α_t at time `t`.
    """

    return exp(log(1-t))
end

function linear(t::Real)

    """Calculates the noise schedule parameter α_t based on the loglinear noise scheduler

        # Arguments
        - `t::Real`: The time step, a real number in the range [0, 1].

        # Returns
        - A real number representing the noise schedule parameter α_t at time `t`.
    """

    return 1 - t
end