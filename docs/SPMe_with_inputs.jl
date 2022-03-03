#
# SPMe with input parameters, comparing with PyBaMM
#

using PyBaMM
using PyCall

# load model
pybamm = pyimport("pybamm")
model = pybamm.lithium_ion.SPMe(name="SPMe")
parameter_values = model.default_parameter_values
parameter_values.update(
    Dict(
        "Negative particle radius [m]" => pybamm.InputParameter("R_n") * 1e-5,
        "Positive particle radius [m]" => pybamm.InputParameter("R_p") * 1e-5
    )
)
sim = pybamm.Simulation(model, parameter_values=parameter_values)

using OrderedCollections
inputs = OrderedDict{String,Float64}(
    "R_n" => 0.5,
    "R_p" => 0.5
)

prob = get_ode_problem(sim, inputs)

using Sundials
sol = solve(prob, CVODE_BDF(linear_solver=:Band,jac_upper=1,jac_lower=1), reltol=1e-6, abstol=1e-6, saveat=prob.tspan[2] / 100);

# Calculate voltage in Julia
V = get_variable(sim, sol, "Terminal voltage [V]", inputs)
t = get_variable(sim, sol, "Time [s]")

# Solve in python
sol_pybamm = sim.solve(t, inputs=inputs)
V_pybamm = get(sol_pybamm, "Terminal voltage [V]").data

# Plots
using Plots

plot(sol.t, V)
scatter!(sol.t, V_pybamm)