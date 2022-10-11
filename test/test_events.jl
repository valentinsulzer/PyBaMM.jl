using PyBaMM
using SparseArrays, LinearAlgebra
using OrdinaryDiffEq
using Test

pybamm = pyimport("pybamm")
# load model
@testset "Test Event Handling"  begin 
    model = pybamm.lithium_ion.SPMe(name="SPMe")
    sim = pybamm.Simulation(model)
    prob,cbs = get_ode_problem(sim)
    prob = remake(prob,tspan=(0,0.5))
    event_to_test = sim.built_model.events[3]
    problem_size = length(prob.u0)
    sol = solve(prob,Rodas5(autodiff=false),callback=cbs)
    @test sol.retcode==:Terminated
end
