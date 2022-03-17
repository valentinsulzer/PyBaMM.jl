
using PyBaMM
using SparseArrays, LinearAlgebra
using PyCall
using DifferentialEquations

# load model
pybamm = pyimport("pybamm")
model = pybamm.lithium_ion.SPMe(name="SPMe")
sim = pybamm.Simulation(model)
prob = get_ode_problem(sim)
prob = remake(prob,tspan=(0,0.5))
event_to_test = sim.built_model.events[8]
cb = build_callback(event_to_test,101)
sol = solve(prob,Rodas5(autodiff=false),callback=cb)
