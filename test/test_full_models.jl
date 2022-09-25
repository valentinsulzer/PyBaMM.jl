using Test
using PyBaMM
using SparseArrays, LinearAlgebra
using Sundials
using OrdinaryDiffEq

pybamm = pyimport("pybamm")

@testset "Compare with PyBaMM: SPM" begin
    # load model
    model = pybamm.lithium_ion.SPM(name="SPM")
    sim = pybamm.Simulation(model)

    prob,cbs = get_ode_problem(sim)

    sol = solve(prob, CVODE_BDF(), saveat=prob.tspan[2] / 100);

    # Calculate voltage in Julia
    V = get_variable(sim, sol, "Terminal voltage [V]")
    t = get_variable(sim, sol, "Time [s]")

    # Solve in python
    sol_pybamm = sim.solve(t)
    V_pybamm = pyconvert(Array{Float64},get(sol_pybamm, "Terminal voltage [V]",nothing).data)
    @test all(isapprox.(V_pybamm, V, atol=1e-3))
end

@testset "Compare with PyBaMM: SPMe" begin
    # load model
    model = pybamm.lithium_ion.SPMe(name="SPMe")
    sim = pybamm.Simulation(model)

    prob,cbs = get_ode_problem(sim)

    sol = solve(prob, CVODE_BDF(), saveat=prob.tspan[2] / 100);

    # Calculate voltage in Julia
    V = get_variable(sim, sol, "Terminal voltage [V]")
    t = get_variable(sim, sol, "Time [s]")

    # Solve in python
    sol_pybamm = sim.solve(t)
    V_pybamm = pyconvert(Array{Float64},get(sol_pybamm, "Terminal voltage [V]",nothing).data)
    @test all(isapprox.(V_pybamm, V, atol=1e-4))
end

@testset "Compare with PyBaMM: DFN" begin
    # load model
    model = pybamm.lithium_ion.DFN(name="DFN")
    sim = pybamm.Simulation(model)

    prob,cbs = get_dae_problem(sim)

    sol = solve(prob, IDA(), saveat=prob.tspan[2] / 100);

    # Calculate voltage in Julia
    V = get_variable(sim, sol, "Terminal voltage [V]")
    t = get_variable(sim, sol, "Time [s]")

    # Solve in python
    sol_pybamm = sim.solve(t)
    V_pybamm = pyconvert(Array{Float64},get(sol_pybamm, "Terminal voltage [V]",nothing).data)
    @test all(isapprox.(V_pybamm, V, atol=1e-3))
end


@testset "Compare with PyBaMM: Semi-Explicit DFN" begin
    # load model
    model = pybamm.lithium_ion.DFN(name="DFN")
    sim = pybamm.Simulation(model)

    prob,cbs = get_dae_problem(sim,dae_type="semi-explicit")

    sol = solve(prob, Rodas4(autodiff=false), saveat=prob.tspan[2] / 100,reltol=1e-10);

    # Calculate voltage in Julia
    V = get_variable(sim, sol, "Terminal voltage [V]")
    t = get_variable(sim, sol, "Time [s]")

    # Solve in python
    sol_pybamm = sim.solve(t)
    V_pybamm = pyconvert(Array{Float64},get(sol_pybamm, "Terminal voltage [V]",nothing).data)
    @test all(isapprox.(V_pybamm, V, atol=1e-3))
end
