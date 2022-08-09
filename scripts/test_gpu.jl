using PyCall,PyBaMM, OrdinaryDiffEq
using GeneralizedGenerated
using LinearAlgebra, SparseArrays, CUDA,ModelingToolkit

pybamm = pyimport("pybamm")
#pyimport("pybamm")."reload"(pybamm)
var_pts = Dict(
    "R_n"=>10,
    "R_p"=>10,
    "r_p"=>10,
    "r_n"=>10,
    "x_p"=>10,
    "x_n"=>10,
    "z"=>10,
    "x_s"=>10,
    "y"=>10
)
model = pybamm.lithium_ion.DFN(name="DFN")
sim = pybamm.Simulation(model)

prob,cbs = get_dae_problem(sim,cache_type="symbolic",dae_type="semi-explicit")

sys = modelingtoolkitize(prob)
prob_jac = ODEProblem(sys,[],prob.tspan,jac=true)

sol = solve(sys, Rodas5(), saveat=prob.tspan[2] / 100,reltol=1e-10);

using BenchmarkTools
@btime solve(sys,Rodas5(),save_everystep=false)
