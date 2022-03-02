module PyBaMM

using GeneralizedGenerated
using LinearAlgebra, SparseArrays
using PyCall

include("simulations.jl")
export get_ode_problem

include("variables.jl")
export get_variable

end # module