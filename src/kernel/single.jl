using Distributions
using LinearAlgebra

include("./type.jl")


function norm(x1, x2; p=2)
    abs(x1 - x2)^p
end
struct Noise{T} <: SingleKernel
    a::T
end

function (nk::Noise)(x1, x2)
    nk.a * (x1 == x2)
end

struct Gaussian{T1,T2} <: SingleKernel
    a::T1
    b::T2
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

function Base.:*(kernel::SingleKernel...)
    ProductKernel(kernel...)
end

function Base.:*(k1::SingleKernel, k2::ProductKernel)
    push!(k2.elements, k1)
end

function Base.:+(kernel::SingleKernel...)
    SumKernel(kernel...)
end
