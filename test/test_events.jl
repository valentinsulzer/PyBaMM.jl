using PyBaMM
using SparseArrays, LinearAlgebra
using PyCall
using DifferentialEquations
using Test

# load model
@testset "Test Event Handling" begin
    pybamm = pyimport("pybamm")
    model = pybamm.lithium_ion.SPMe(name="SPMe")
    sim = pybamm.Simulation(model)
    prob = get_ode_problem(sim)
    prob = remake(prob,tspan=(0,0.5))
    event_to_test = sim.built_model.events[8]
    problem_size = length(prob.u0)
    cb = build_callback(event_to_test,problem_size)
    sol = solve(prob,Rodas5(autodiff=false),callback=cb)
    @test sol.retcode==:terminated
end
