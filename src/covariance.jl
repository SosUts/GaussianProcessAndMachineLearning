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