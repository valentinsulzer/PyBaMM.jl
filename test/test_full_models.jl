using Test

@testset "SPM" begin
    
    @test sol_pybamm. .≈ sol.u
    @test V_pybamm .≈ V
end