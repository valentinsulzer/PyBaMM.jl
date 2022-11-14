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

    for model in [spm, spme, dfn]
        sim = pybamm.Simulation(model)
        sim.build()
        prob,cbs = get_dae_problem(sim,dae_type="semi-explicit", cache_type = "dual")
        jac_fn! = generate_jacobian(sim)
        fn! = prob.f.f
        len_y = convert(Int, pyconvert(Int,sim.built_model.len_rhs_and_alg))
        p=nothing
        u0 = prob.u0
        function myjac(u0)
            du = similar(u0)
            fn!(du, u0, nothing, 0.0)
            return du
        end
        fd_jac = ForwardDiff.jacobian(myjac,u0)

        pybamm_jac = similar(fd_jac)
        jac_fn!(pybamm_jac, u0, nothing, 0.0)
        
        @test all(isapprox.(pybamm_jac,fd_jac, atol=1e-3))
    end
end
