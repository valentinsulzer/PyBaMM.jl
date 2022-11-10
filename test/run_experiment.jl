using PyBaMM
pybamm = pyimport("pybamm")

options = 

model = pybamm.lithium_ion.DFN(name="DFN")
experiment = pybamm.Experiment([
    "Discharge at 54W for 75 sec",
    "Discharge at 16W for 800 sec",
    "Discharge at 54W for 75 sec",
    "Rest for 600 sec",
    "Charge at 1 C until 4.2 V",
    "Hold at 4.2 V until 500 mA",
    "Rest for 600 sec",
])

parameters = pybamm.ParameterValues("Chen2020")
parameters["Nominal cell capacity [A.h]"] = 3.0

sim = pybamm.Simulation(model, experiment=experiment, parameter_values=parameters)

include("../src/experiments.jl")


vars_to_save = ["Terminal voltage [V]", "Cell temperature [K]"]

sols = solve_with_experiment(sim, vars_to_save = vars_to_save, save_everystep=false)

using Plots
plotly()

plot(sols[3], sols[2]["Terminal voltage [V]"])
