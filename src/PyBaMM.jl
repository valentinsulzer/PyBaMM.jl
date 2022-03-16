module PyBaMM

using GeneralizedGenerated
using LinearAlgebra, SparseArrays
using PyCall

include("diffeq_problems.jl")
export get_ode_problem, get_dae_problem

include("variables.jl")
export get_variable, get_l2loss_function

end # module