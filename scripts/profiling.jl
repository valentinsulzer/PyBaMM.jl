using GeneralizedGenerated
using LinearAlgebra, SparseArrays
using PyCall
using Profile
using PProf


pybamm = pyimport("pybamm")
tend=3600
inputs=nothing

model = pybamm.lithium_ion.DFN(name="DFN")
sim = pybamm.Simulation(model)

input_parameter_order = isnothing(inputs) ? nothing : collect(keys(inputs))
    p = isnothing(inputs) ? nothing : collect(values(inputs))
    
    sim.build()
    fn_str, u0_str = sim.built_model.generate_julia_diffeq(
        input_parameter_order=input_parameter_order, 
        dae_type="implicit", 
        get_consistent_ics_solver=pybamm.CasadiSolver()
    )
    
    # PyBaMM-generated functions
    sim_fn! = runtime_eval(Meta.parse(fn_str))
    sim_u0! = runtime_eval(Meta.parse(u0_str))

    # Evaluate initial conditions
    len_y = convert(Int, sim.built_model.len_rhs_and_alg)
    u0 = Array{Float64}(undef, len_y)
    sim_u0!(u0, p)

    # Scale the time
    tau = sim.built_model.timescale.evaluate()
    tspan = (0, tend/tau)
    
    du = similar(u0)
    out = similar(u0)

    sim_fn!(out,du,u0,nothing,1.2)
