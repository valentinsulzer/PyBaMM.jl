using Test
using PyBaMM
using SparseArrays, LinearAlgebra
using Sundials
using OrderedCollections

pybamm = pyimport("pybamm")

@testset "SPMe L2 loss voltage" begin
    # load model
    model = pybamm.lithium_ion.SPMe(name="SPMe")
    parameter_values = model.default_parameter_values
    parameter_values.update(
        PyDict(Dict(
            "Cation transference number" => pybamm.InputParameter("t_plus"),
            "Electrolyte conductivity [S.m-1]" => pybamm.InputParameter("kappa_e"),
        ))
    )
    sim = pybamm.Simulation(model, parameter_values=parameter_values)

    inputs = OrderedDict{String,Float64}(
        "t_plus" => 0.4,
        "kappa_e" => 0.7,
    )

    # Build ODE problem
    prob,cbs = get_ode_problem(sim, inputs)
    t = prob.tspan[2] / 100
    # Generate data
    sol = solve(prob, CVODE_BDF(linear_solver=:Band,jac_upper=1,jac_lower=1), reltol=1e-6, abstol=1e-6, saveat=t);
    V_data = get_variable(sim, sol, "Terminal voltage [V]", inputs)

    # Get loss function
    loss = get_l2loss_function(sim, "Terminal voltage [V]", inputs, V_data)
    
    # Loss of solution used to generate data should be zero
    @test loss(sol) == 0.0

    # Loss should get bigger as parameters get further away from truth
    prob2 = remake(prob; p=[0.5,0.7])
    sol2 = solve(prob2, CVODE_BDF(linear_solver=:Band,jac_upper=1,jac_lower=1), reltol=1e-6, abstol=1e-6, saveat=t);
    
    prob3 = remake(prob; p=[0.6,0.7])
    sol3 = solve(prob3, CVODE_BDF(linear_solver=:Band,jac_upper=1,jac_lower=1), reltol=1e-6, abstol=1e-6, saveat=t);
    @test loss(sol2) < loss(sol3)
end



@testset "DFN L2 loss voltage" begin
    # load model
    model = pybamm.lithium_ion.DFN(name="DFN")
    parameter_values = model.default_parameter_values
    parameter_values.update(
        PyDict(Dict(
            "Cation transference number" => pybamm.InputParameter("t_plus"),
            "Electrolyte conductivity [S.m-1]" => pybamm.InputParameter("kappa_e"),
        ))
    )
    sim = pybamm.Simulation(model, parameter_values=parameter_values)

    inputs = OrderedDict{String,Float64}(
        "t_plus" => 0.4,
        "kappa_e" => 0.7,
    )

    # Build ODE problem
    prob,cbs = get_dae_problem(sim, inputs,dae_type="semi-explicit",cache_type="dual")
    t = prob.tspan[2] / 100
    # Generate data
    sol = solve(prob, ROS34PW2(autodiff=false), reltol=1e-6, abstol=1e-6, saveat=t);
    V_data = get_variable(sim, sol, "Terminal voltage [V]", inputs)

    # Get loss function
    loss = get_l2loss_function(sim, "Terminal voltage [V]", inputs, V_data)
    
    # Loss of solution used to generate data should be zero
    @test loss(sol) == 0.0

    # Loss should get bigger as parameters get further away from truth
    prob2 = remake(prob; p=[0.5,0.7])
    sol2 = solve(prob2, ROS34PW2(autodiff=false), reltol=1e-6, abstol=1e-6, saveat=t);
    
    prob3 = remake(prob; p=[0.6,0.7])
    sol3 = solve(prob3, ROS34PW2(autodiff=false), reltol=1e-6, abstol=1e-6, saveat=t);
    @test loss(sol2) < loss(sol3)
end
