#
# Solving the SPMe and comparison with the PyBaMM solution
#

using PyBaMM
using SparseArrays, LinearAlgebra
using PyCall

# load model
pybamm = pyimport("pybamm")
model = pybamm.lithium_ion.SPMe(name="SPMe")
sim = pybamm.Simulation(model)

prob,cbs = get_ode_problem(sim)

using Sundials
sol = solve(prob, CVODE_BDF(linear_solver=:Band,jac_upper=1,jac_lower=1), reltol=1e-6, abstol=1e-6, saveat=prob.tspan[2] / 100);

# Calculate voltage in Julia
V = get_variable(sim, sol, "Terminal voltage [V]")
t = get_variable(sim, sol, "Time [s]")

# Solve in python
sol_pybamm = sim.solve(t)
V_pybamm = get(sol_pybamm, "Terminal voltage [V]").data

# Plots
using Plots

plot(sol.t, V)
scatter!(sol.t, V_pybamm)