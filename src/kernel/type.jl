abstract type Kernel end

abstract type SingleKernel <: Kernel end
abstract type Matern <: SingleKernel end

abstract type CombinationKernel <: Kernel end
