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
    )
)
sim = pybamm.Simulation(model, parameter_values=parameter_values)

using OrderedCollections
inputs = OrderedDict{String,Float64}(
    "R_n" => 0.5,
)

prob = get_ode_problem(sim, inputs)

# Generate data
using Sundials
sol = solve(prob, CVODE_BDF(linear_solver=:Band,jac_upper=1,jac_lower=1), reltol=1e-6, abstol=1e-6, saveat=prob.tspan[2] / 100);
V_data = get_variable(sim, sol, "Terminal voltage [V]", inputs)
t = get_variable(sim, sol, "Time [s]")

# Calculate voltage in Julia
using DiffEqParamEstim
cost_function = build_loss_objective(prob,CVODE_BDF(linear_solver=:Band,jac_upper=1,jac_lower=1),L2Loss(t,data),
                                     maxiters=10000,verbose=false)