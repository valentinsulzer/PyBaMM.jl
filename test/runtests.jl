using PyBaMM
using Test
using SafeTestsets
using SparseArrays, LinearAlgebra

@safetestset "Compare models with PyBaMM" begin include("test_full_models.jl") end
@safetestset "Test loss functions" begin include("test_loss.jl") end