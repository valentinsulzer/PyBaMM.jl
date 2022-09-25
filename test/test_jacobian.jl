using Test
using PyBaMM

pybamm = pyimport("pybamm")

@testset "Compare Jacobians" begin
    # load model
    spm = pybamm.lithium_ion.SPM(name="SPM")
    spme = pybamm.lithium_ion.SPMe(name="SPMe")
    dfn = pybamm.lithium_ion.DFN(name="DFN")

    input_parameter_order = []
    dae_type = "semi-explicit"
    get_consistent_ics_solver=pybamm.CasadiSolver()
    preallocate=true
    cache_type="dual"
    generate_jacobian=true

    for model in [dfn]
        sim = pybamm.Simulation(model)
        sim.build()
        
        # Get the jacobian
        fn_str, u0, jac_str = sim.built_model.generate_julia_diffeq(
            input_parameter_order=input_parameter_order,
            dae_type=dae_type,
            get_consistent_ics_solver=get_consistent_ics_solver,
            preallocate=preallocate,
            cache_type=cache_type,
            generate_jacobian=generate_jacobian
        )
        u0 = vec(pyconvert(Array{Float64},u0.evaluate()))
        fn! = runtime_eval(Meta.parse(pyconvert(String,fn_str)))
        jac_fn! = runtime_eval(Meta.parse(pyconvert(String,jac_str)))
        function myjac(u0)
            du = similar(u0)
            fn!(du, u0, nothing, 0.0)
            return du
        end

        fn! = runtime_eval(Meta.parse(pyconvert(String,fn_str)))
        fd_jac = ForwardDiff.jacobian(myjac,u0)

        pybamm_jac = similar(fd_jac)
        jac_fn!(pybamm_jac, u0, nothing, 0.0)
        
        @test all(isapprox.(pybamm_jac,fd_jac, atol=1e-3))
    end
end
