
using Plots

# Number of samples
N = 100

function random_float_in_interval(i, j)
    return i + (j - i) * rand()
end

# Function to generate low-discrepancy samples
function low_discrepancy_samples(N)
    samples = [random_float_in_interval((i-1)/N, i/N) for i in 1:N]
    return samples
end

# Generate samples
samples = low_discrepancy_samples(N)
og_samples = 0:0.01:1

# Plot the samples
scatter([samples, og_samples], zeros(N), label="Samples", xlabel="x", ylabel="", title="Low Discrepancy Samples", markersize=5)
hline!([0], label="", line=:solid)  # Add a horizontal line at y=0 for visual clarity

# Show the plot
display(plot)