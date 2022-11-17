module PyBaMM


using Reexport
@reexport using GeneralizedGenerated
@reexport using CUDA
@reexport using LinearSolve
@reexport using LinearAlgebra, SparseArrays
@reexport using Symbolics
@reexport using PreallocationTools
@reexport using ModelingToolkit
@reexport using PythonCall
@reexport using ForwardDiff
@reexport using OrdinaryDiffEq
@reexport using OrderedCollections
@reexport using Sundials

const pybamm = PythonCall.pynew()
const pack = PythonCall.pynew()
const pybamm2julia = PythonCall.pynew()
const setup_circuit = PythonCall.pynew()
const sys = PythonCall.pynew()
const pycopy = PythonCall.pynew()

function __init__()
    PythonCall.pycopy!(sys, pyimport("sys"))
    sys.path.append(joinpath(@__DIR__,"../pysrc"))
    PythonCall.pycopy!(pybamm, pyimport("pybamm"))
    PythonCall.pycopy!(pack, pyimport("pack"))
    PythonCall.pycopy!(pybamm2julia, pyimport("pybamm2julia"))
    PythonCall.pycopy!(setup_circuit, pyimport("setup_circuit"))
    PythonCall.pycopy!(pycopy, pyimport("copy"))
end

include("diffeq_problems.jl")
export get_ode_problem, get_dae_problem,get_semiexplicit_dae_problem,get_optimized_problem

include("variables.jl")
export get_variable, get_l2loss_function

include("events.jl")
export build_callback

include("symcache.jl")
export DiffCache,get_tmp,symcache

include("jacobian.jl")
export generate_jacobian

include("pack_postprocessing.jl")
export get_pack_variables

end # module