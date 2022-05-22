using Distributions
using LinearAlgebra
using Tullio
using PyPlot
using PDMats


abstract type Kernel end

abstract type SingleKernel end

struct Gaussian{T1,T2} <: SingleKernel
    a::T1
    b::T2
end

function norm(x1, x2; p=2)
    abs(x1 - x2)^p
end

function (gk::Gaussian)(x1, x2)
    gk.a * exp(-norm(x1, x2; p=2) / gk.b)
end

struct Linear <: SingleKernel end

function (::Linear)(x1, x2)
    x1 * x2 + 1.0
end

struct Exponential{T} <: SingleKernel
    a::T
end

function (ek::Exponential)(x1, x2)
    exp(-abs(x1 - x2) / ek.a)
end

struct Periodic{T1,T2} <: SingleKernel
    a::T1
    b::T2
end

function (pk::Periodic)(x1, x2)
    exp(pk.a * cos(abs(x1 - x2) / pk.b))
end

abstract type Matern <: SingleKernel end

struct Matern3{T} <: SingleKernel
    a::T
    nu
    Matern3(a::T) where {T} = new{T}(a, 1.5)
end

function (m3k::Matern3)(x1, x2)
    r = norm(x1, x2, p=1)
    (1 + r * sqrt(3) / m3k.a) * exp(-r * sqrt(3) / m3k.a)
end

struct Matern5{T} <: Matern
    a::T
    nu
    Matern3(a::T) where {T} = new{T}(a, 2.5)
end

function (m5k::Matern5)(x1, x2)
    r = norm(x1, x2, p=1)
    (1 + r * sqrt(5) / m5k.a + 5 * abs2(r) / (3 * abs2(m5k.a))) * exp(-r * sqrt(5) / m5k.a)
end

abstract type CombinationKernel <: Kernel end

struct SumKernel <: CombinationKernel
    kernels::Vector{SingleKernel}
    function SumKernel(ks...)
        kernels = Vector{SingleKernel}(undef, length(ks))
        @tullio kernels[i] = ks[i]
        new(kernels)
    end
end

struct ProductKernel <: CombinationKernel
    kernels::Vector{SingleKernel}
    function ProductKernel(ks...)
        kernels = Vector{SingleKernel}(undef, length(ks))
        @tullio kernels[i] = ks[i]
        new(kernels)
    end
end


function K(kernel::SingleKernel, x1, x2)
    length(x1) == length(x2) || throw(DomainError("Inputs must have same length"))
    K = Array{promote_type(eltype(x1), eltype(x2))}(undef, length(x1), length(x1))
    @tullio K[i, j] = kernel(x1[i], x2[j])
end

function K(kernel::SingleKernel, x)
    K = Array{eltype(x)}(undef, length(x), length(x))
    @tullio K[i, j] = kernel(x[i], x[j])
end

function K(combkernels::CombinationKernel, x1, x2)
    length(x1) == length(x2) || throw(DomainError("Inputs must have same length"))
    K = Array{promote_type(eltype(x1), eltype(x2))}(undef, length(x1), length(x1))
    @tullio K[i, j] = combkernels.kernels[k](x1[i], x2[j])
end

function K(combkernels::CombinationKernel, x1)
    K = Array{eltype(x)}(undef, length(x), length(x))
    @tullio K[i, j] = combkernels.kernels[k](x[i], x[j])
end

function Base.:*(k1::SingleKernel, k2::SingleKernel)
    ProductKernel(k1, k2)
end

function Base.:*(k1::ProductKernel, k2::SingleKernel)
    push!(k1.kernels, k2)
end

function Base.:*(k1::SingleKernel, k2::ProductKernel)
    push!(k2.kernels, k1)
end

function Base.:+(k1::SingleKernel, k2::SingleKernel)
    SumKernel(k1, k2)
end

function Base.:+(k1::SumKernel, k2::SingleKernel)
    push!(k1.kernels, k2)
end

function Base.:+(k1::SingleKernel, k2::ProductKernel)
    push!(k2.kernels, k1)
end

"""
https://github.com/STOR-i/GaussianProcesses.jl/blob/9fe174a1753796fe7295b81d76536e5eb1c37f32/src/GP.jl#L91-L116
"""
function make_posdef!(m::AbstractMatrix, chol_factors::AbstractMatrix; nugget=1.0e-10)
    n = size(m, 1)
    size(m, 2) == n || throw(ArgumentError("Covariance matrix must be square"))
    if nugget > 0
        @inbounds for i in 1:n
            m[i, i] += nugget
        end
    end
    copyto!(chol_factors, m)
    chol = cholesky!(Symmetric(chol_factors, :U))
    m, chol
end

function make_posdef!(m::AbstractMatrix; nugget=1.0e-10)
    chol_buffer = similar(m)
    make_posdef!(m, chol_buffer; nugget=nugget)
end