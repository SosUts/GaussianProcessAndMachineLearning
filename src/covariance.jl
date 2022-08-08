using Distributions
using LinearAlgebra
using PyPlot
using PDMats

include("./kernel/single.jl")
include("./kernel/composite.jl")

function _make_posdef!(m::AbstractMatrix, chol_factors::AbstractMatrix; nugget=1.0e-10)
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

function _make_posdef!(m::AbstractMatrix; nugget=1.0e-10)
    """
    https://github.com/STOR-i/GaussianProcesses.jl/blob/9fe174a1753796fe7295b81d76536e5eb1c37f32/src/GP.jl#L91-L116
    """
    chol_buffer = similar(m)
    _make_posdef!(m, chol_buffer; nugget=nugget)
end

function cov(kernel::Kernel, x1, x2)
    kernel.(x1, x2')
end

function cov(kernel::Kernel, x)
    cov(kernel, x, x)
end


function predict(kernel, train_x, train_y, predict_x)
    inv_Ktt = pinv(cov(kernel, train_x))
    ktp = cov(kernel, train_x, predict_x)
    kpp = cov(kernel, predict_x)

    trained_mu = ktp' * inv_Ktt * train_y
    trained_cov = kpp - ktp' * inv_Ktt * ktp
    try
        return MvNormal(trained_mu, Symmetric(trained_cov))
    catch
        return MvNormal(trained_mu, Symmetric(_make_posdef!(trained_cov)[1]))
    end
end
