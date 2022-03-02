#
# Create and discretize the SPMe in pybamm, convert it to a format Julia DiffEq likes, and solve
# A lot of this could eventually be converted into functions in PyBaMM.jl
#

using PyBaMM
using PyCall

pybamm = pyimport("pybamm")

# load model
model = pybamm.lithium_ion.SPMe(name="SPMe")
sim = pybamm.Simulation(model)

prob = get_ode_problem(sim)

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