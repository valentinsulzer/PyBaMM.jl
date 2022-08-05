using PyCall,PyBaMM, OrdinaryDiffEq
using GeneralizedGenerated
using LinearAlgebra, SparseArrays

pybamm = pyimport("pybamm")

model = pybamm.lithium_ion.DFN(name="DFN")
sim = pybamm.Simulation(model)
inputs=nothing
dae_type="semi-explicit"

input_parameter_order = isnothing(inputs) ? nothing : collect(keys(inputs))
    p = isnothing(inputs) ? nothing : collect(values(inputs))
    
    sim.build()
    fn_str, u0_str = sim.built_model.generate_julia_diffeq(
        input_parameter_order=input_parameter_order, 
        dae_type=dae_type, 
        get_consistent_ics_solver=pybamm.CasadiSolver(),
        preallocate=true,
        cache_type="symbolic"
    )

io = open("newfile.jl","w")
println(io,fn_str)
close(io)