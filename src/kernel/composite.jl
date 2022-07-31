using Distributions
using LinearAlgebra

include("./single.jl")


struct SumKernel <: CombinationKernel
    elements::Vector{SingleKernel}
    function SumKernel(ks...)
        kernels = Vector{SingleKernel}(undef, length(ks))
        @tullio kernels[i] = ks[i]
        new(kernels)
    end
end

struct ProductKernel <: CombinationKernel
    elements::Vector{SingleKernel}
    function ProductKernel(ks...)
        kernels = Vector{SingleKernel}(undef, length(ks))
        @tullio kernels[i] = ks[i]
        new(kernels)
    end
end

function (sk::SumKernel)(x1, x2)
    value = 0.0
    for element in sk.elements
        value += element(x1, x2)
    end
    value
end

function Base.:*(k1::ProductKernel, k2::SingleKernel...)
    push!(k1.elements, k2...)
end

function Base.:+(k1::SumKernel, k2::SingleKernel)
    push!(k1.elements, k2)
end

function Base.:+(k1::SingleKernel, k2::ProductKernel)
    push!(k2.elements, k1)
end

function Base.:iterate(combination_kernels::CombinationKernel, state=1)
    if state > length(combination_kernels.elements)
        return nothing
    else
        return (combination_kernels.elements[state], state + 1)
    end
end