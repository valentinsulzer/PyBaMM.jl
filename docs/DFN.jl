#
# DFN (work in progress)
#

using PyBaMM
using PyCall
using SparseArrays, LinearAlgebra

pybamm = pyimport("pybamm")
np = pyimport("numpy")

# load model
model = pybamm.lithium_ion.DFN(name="DFN")

sim = pybamm.Simulation(model)
prob,cbs = get_dae_problem(sim, 3600, nothing)

using Sundials
sol = solve(prob, IDA());#, reltol=1e-6, abstol=1e-6, saveat=tend / 100);
sol.u

# Benchmarks
using BenchmarkTools
@btime solve(prob, IDA());

# Calculate voltage in Julia
V = get_variable(sim, sol, "Terminal voltage [V]")

# Solve in python
sol_pybamm = sim.solve(sol.t * sim.built_model.timescale.evaluate())
V_pybamm = get(sol_pybamm, "Terminal voltage [V]").data

# Plots
using Plots

plot(sol.t, V)
scatter!(sol.t, V_pybamm)