#using PyCall,PyBaMM,BenchmarkTools,DifferentialEquations,SparseArrays,LinearSolve,Symbolics,IncompleteLU,GeneralizedGenerated,CUDA
using PyBaMM
using IncompleteLU
using OrderedCollections
using DiffEqSensitivity
using Zygote


pybamm = pyimport("pybamm")
model = pybamm.lithium_ion.DFN(name="DFN")

var_pts = Dict(
    "R_n"=>10,
    "R_p"=>10,
    "r_p"=>10,
    "r_n"=>10,
    "x_p"=>10,
    "x_n"=>10,
    "z"=>10,
    "x_s"=>10,
    "y"=>10
)
parameter_values = model.default_parameter_values
parameter_values.update(
    PyDict(Dict(
        "Negative electrode diffusivity [m2.s-1]" => pybamm.InputParameter("t_plus"),
        "Electrolyte conductivity [S.m-1]" => pybamm.InputParameter("kappa_e"),
    ))
)
sim = pybamm.Simulation(model, parameter_values=parameter_values, var_pts=var_pts)
sim.build()

inputs = OrderedDict{String,Float64}(
    "t_plus" => 1e-5,
    "kappa_e" => 0.7,
)

prob,cbs = get_dae_problem(sim,inputs,dae_type="semi-explicit",cache_type="dual")

u0 = deepcopy(prob.u0)
du0 = similar(u0)
p = prob.p
t = 0.0
prob.f(du0,u0,p,t)

alg = QBDF

sol = solve(prob,alg())

outer_p = vcat(prob.p,prob.u0)

function my_sensitivity(new_p)
    prob_local = remake(prob,p=new_p)
    sol = solve(prob_local,alg(autodiff=false))
    return sum(sum(Array(sol)))
end

ForwardDiff.gradient(my_sensitivity,prob.p)
