#
# SPMe with input parameters, comparing with PyBaMM
#

using PyBaMM
using SparseArrays, LinearAlgebra
using PyCall

# load model
pybamm = pyimport("pybamm")
model = pybamm.lithium_ion.SPMe(name="SPMe")
parameter_values = model.default_parameter_values
parameter_values.update(
    Dict(
        "Cation transference number" => pybamm.InputParameter("t_plus"),
        "Electrolyte conductivity [S.m-1]" => pybamm.InputParameter("kappa_e"),
    )
)
sim = pybamm.Simulation(model, parameter_values=parameter_values)

using OrderedCollections
inputs = OrderedDict{String,Float64}(
    "t_plus" => 0.4,
    "kappa_e" => 0.7,
)

# Build ODE problem
prob,cbs = get_ode_problem(sim, inputs)

# Generate data
using Sundials
sol = solve(prob, CVODE_BDF(linear_solver=:Band,jac_upper=1,jac_lower=1), reltol=1e-6, abstol=1e-6, saveat=prob.tspan[2] / 100);
V_data = get_variable(sim, sol, "Terminal voltage [V]", inputs) + 0.005*randn(size(sol.t))
t = get_variable(sim, sol, "Time [s]")

using Plots
plot(t,V_data)

# Build loss objective
using DiffEqParamEstim

loss = get_l2loss_function(sim, "Terminal voltage [V]", inputs, V_data)
cost_function = build_loss_objective(prob,CVODE_BDF(linear_solver=:Band,jac_upper=1,jac_lower=1),loss,
                                     maxiters=10000,verbose=true,saveat=sol.t)
                                 
using BenchmarkTools
@btime loss(sol)
@btime get_variable(sim, sol, "Terminal voltage [V]", inputs)
# Sanity check: plot 1-parameter cost function
vals = 0.1:0.01:0.8
cost = [cost_function([0.4,i]) for i in vals]
using Plots; plotly()
plot(vals,cost,yscale=:log10,
    xaxis = "Parameter", yaxis = "Cost", title = "1-Parameter Cost Function",
    lw = 3)

# Optimize
using Optim
result = optimize(cost_function, [0.3, 0.6], BFGS())
@show result.minimizer