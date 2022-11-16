using Test
using PyBaMM
using SparseArrays, LinearAlgebra
using Sundials
using OrdinaryDiffEq


pybamm=pyimport("pybamm")

@testset "Dual Cache ODE" begin
    # load model
    model = pybamm.lithium_ion.SPM(name="SPM")
    sim = pybamm.Simulation(model)

    prob,cbs = get_ode_problem(sim,cache_type="dual")

    sol = solve(prob, Rodas5(), saveat=prob.tspan[2] / 100);

    # Calculate voltage in Julia
    V = get_variable(sim, sol, "Terminal voltage [V]")
    t = get_variable(sim, sol, "Time [s]")

    # Solve in python
    sol_pybamm = sim.solve(t)
    V_pybamm = pyconvert(Array{Float64},get(sol_pybamm, "Terminal voltage [V]",nothing).data)
    @test all(isapprox.(V_pybamm, V, atol=1e-2))
end

@testset "Symbolic Cache ODE" begin
    # load model
    model = pybamm.lithium_ion.SPM(name="SPM")
    sim = pybamm.Simulation(model)

    prob,cbs = get_ode_problem(sim,cache_type="symbolic")

    sys = modelingtoolkitize(prob)
    prob = ODEProblem(sys,Pair[],prob.tspan,jac=true,sparse=true)
    sol = solve(prob, Rodas5(), saveat=prob.tspan[2] / 100);

    # Calculate voltage in Julia
    V = get_variable(sim, sol, "Terminal voltage [V]")
    t = get_variable(sim, sol, "Time [s]")

    # Solve in python
    sol_pybamm = sim.solve(t)
    V_pybamm = pyconvert(Array{Float64},get(sol_pybamm, "Terminal voltage [V]",nothing).data)
    @test all(isapprox.(V_pybamm, V, atol=1e-2))
end


@testset "Dual Cache Semi-Explicit DAE" begin
    # load model
    model = pybamm.lithium_ion.DFN(name="DFN")
    sim = pybamm.Simulation(model)

    prob,cbs = get_dae_problem(sim,cache_type="dual",dae_type="semi-explicit")

    sol = solve(prob, ROS34PW2(), saveat=prob.tspan[2] / 100,reltol=1e-7);

    # Calculate voltage in Julia
    V = get_variable(sim, sol, "Terminal voltage [V]")
    t = get_variable(sim, sol, "Time [s]")

    # Solve in python
    sol_pybamm = sim.solve(t)
    V_pybamm = pyconvert(Array{Float64},get(sol_pybamm, "Terminal voltage [V]",nothing).data)
    @test all(isapprox.(V_pybamm, V, atol=1e-2))
end

@testset "Dual Cache Implicit DAE" begin
    # load model
    model = pybamm.lithium_ion.DFN(name="DFN")
    sim = pybamm.Simulation(model)

    prob,cbs = get_dae_problem(sim,cache_type="dual",dae_type="implicit")

    sol = solve(prob, DFBDF(), saveat=prob.tspan[2] / 100,reltol=1e-7);

    # Calculate voltage in Julia
    V = get_variable(sim, sol, "Terminal voltage [V]")
    t = get_variable(sim, sol, "Time [s]")

    # Solve in python
    sol_pybamm = sim.solve(t)
    V_pybamm = pyconvert(Array{Float64},get(sol_pybamm, "Terminal voltage [V]",nothing).data)
    @test all(isapprox.(V_pybamm, V, atol=1e-2))
end

#=
@testset "Symbolic Cache Semi-Explicit DAE" begin
    # load model
    model = pybamm.lithium_ion.DFN(name="DFN")
    sim = pybamm.Simulation(model)

    prob,cbs = get_dae_problem(sim,cache_type="symbolic",dae_type="semi-explicit")
    sys = modelingtoolkitize(prob)
    prob = ODEProblem(sys,Pair[],prob.tspan,)

    sol = solve(prob, ROS34PW2(), saveat=prob.tspan[2] / 100,reltol=1e-7);

    # Calculate voltage in Julia
    V = get_variable(sim, sol, "Terminal voltage [V]")
    t = get_variable(sim, sol, "Time [s]")

    # Solve in python
    sol_pybamm = sim.solve(t)
    V_pybamm = pyconvert(Array{Float64},get(sol_pybamm, "Terminal voltage [V]",nothing).data)
    @test all(isapprox.(V_pybamm, V, atol=1e-3))
end
=#
#=
COMMENTED PENDING https://github.com/SciML/ModelingToolkit.jl/issues/866
@testset "Symbolic Cache Implicit DAE" begin
    # load model
    model = pybamm.lithium_ion.DFN(name="DFN")
    sim = pybamm.Simulation(model)

    prob,cbs = get_dae_problem(sim,cache_type="symbolic",dae_type="implicit")
    sys = modelingtoolkitize(prob)
    prob = ODEProblem(sys,Pair[],prob.tspan,)

    sol = solve(prob, DFBDF(), saveat=prob.tspan[2] / 100,reltol=1e-7);

    # Calculate voltage in Julia
    V = get_variable(sim, sol, "Terminal voltage [V]")
    t = get_variable(sim, sol, "Time [s]")

    # Solve in python
    sol_pybamm = sim.solve(t)
    V_pybamm = get(sol_pybamm, "Terminal voltage [V]").data
    @test all(isapprox.(V_pybamm, V, atol=1e-3))
end
=#


