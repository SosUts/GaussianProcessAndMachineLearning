include("./single.jl")
include("./composite.jl")

function Base.:show(io::IO, combination_kernel::CombinationKernel)
    type = typeof(combination_kernel)
    println(io, type)
    for kernel in combination_kernel
        println(io, "    ", kernel)
    end
end