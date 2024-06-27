using Test
include("diffusion.jl")

MASK = [0, 0, 0, 1]
model = MaskedDiffusionLanguageModel(3, MASK, cosineschedule)

####### Test 1: Ensure that forward model converts vectorized data correctly ###########

@test forward(model, [[1,0,0,0], [0,1,0,0], [0,0,1,0]], 0, 1) == [MASK, MASK, MASK]

####### Test 2: Ensure that backward model follows the SUBS parameterization ###########

function update_x!(model, x, i)
    global x
    x = backward(model, x, i, i)
end

function mask_indices(ls::AbstractArray)
    indices = Set()
    for (i, value) in ls 
        if value == MASK
            push!(indices, i)
        end
    end
    return indices
end

global x = [MASK, MASK, MASK]

for i in range(0, 1, step=0.1)
    x_before = deepcopy(x)
    update_x!(model, x, i)
    @show x
    # @test length(setdiff(mask_indices(x_before), mask_indices(x))) == 0
    
    for j in x
        if j not in mask_indices(x_before)
            @test x_before[j] == x[j]
        end
    end
end

