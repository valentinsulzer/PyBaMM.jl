using PyBaMM, OrdinaryDiffEq,ModelingToolkit,Symbolics,LinearAlgebra,BenchmarkTools,Sundials,PyCall,SparseArrays

pybamm = pyimport("pybamm")

model = pybamm.lithium_ion.DFN(name="DFN")
sim = pybamm.Simulation(model)

prob,cbs = get_dae_problem(sim,dae_type="semi-explicit")

sys = modelingtoolkitize(prob)
sys2 = structural_simplify(sys)

mtkprob = ODEProblem(sys,[],prob.tspan,jac=true,sparse=true)
mtkprob2 = ODEProblem(sys2,[],prob.tspan,jac=true,sparse=true)
