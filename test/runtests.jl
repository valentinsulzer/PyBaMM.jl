using PyBaMM
using Test
using SafeTestsets

@safetestset "Compare models with PyBaMM" begin include("test_full_models.jl") end