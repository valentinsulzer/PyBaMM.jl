using PyBaMM
pybamm = pyimport("pybamm")

model = pybamm.lithium_ion.DFN(name="DFN")
experiment = pybamm.Experiment(["Discharge at 12W for 75 sec", "Discharge at 3 W for 800 sec","Discharge at 12W for 75 sec","Charge at 1 A until 4.2V", "Hold at 4.2V until 50 mA"])
sim = pybamm.Simulation(model, experiment=experiment)

include("../src/experiments.jl")

probs = get_experiment_probs(sim, nothing, nothing, initial_soc = 1.0)
termination_funcs = get_termination_condition(sim, nothing, nothing)


steps = length(sim.experiment.operating_conditions)

sols = []

for step in 1:steps
    println("doing step $step")
    prob = probs[step]
    term_cond = termination_funcs[step]
    built_model = sim.op_conds_to_built_models[sim.experiment.operating_conditions[step-1]["string"]]
    if step!=1
        #set initial condition for the next step.
        new_model = sim.op_conds_to_built_models[sim.experiment.operating_conditions[step-1]["string"]]
        old_model = sim.op_conds_to_built_models[sim.experiment.operating_conditions[step-2]["string"]]
        new_u0 = deepcopy(prob.u0)
        old_u0 = deepcopy(sols[step-1].u[end])
        set_initial_conditions!(new_u0, old_model, new_model, old_u0)
        prob = remake(prob, u0=new_u0)
    end

    done = false
    integrator = init(prob, Trapezoid(autodiff=false), reltol=1e-3, abstol=1e-3)
    while !done
        step!(integrator)
        done = any(term_cond(integrator.u, integrator.t*pyconvert(Float64, built_model.timescale.evaluate())).<0)
    end
    push!(sols, deepcopy(integrator.sol))
end




        



