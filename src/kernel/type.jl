abstract type Kernel end

abstract type SingleKernel end
abstract type Matern <: SingleKernel end

abstract type CombinationKernel <: Kernel end