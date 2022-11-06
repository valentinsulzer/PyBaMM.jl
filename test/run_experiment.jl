using PyBaMM
pybamm = pyimport("pybamm")

model = pybamm.lithium_ion.DFN(name="DFN")
experiment = pybamm.Experiment(["Charge at 1 A until 4.2V", "Hold at 4.2V until 50 mA"])
sim = pybamm.Simulation(model, experiment=experiment)

sim.build_for_experiment()

