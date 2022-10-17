#using PyCall,PyBaMM,BenchmarkTools,DifferentialEquations,SparseArrays,LinearSolve,Symbolics,IncompleteLU,GeneralizedGenerated,CUDA
using PyBaMM


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


sim = pybamm.Simulation(model,var_pts=var_pts)
sim.build()

#Generate Analytical Jacobian
input_parameter_order = []
dae_type="semi-explicit"
preallocate=true
generate_jacobian=true
cache_type="dual"



fn_str, u0,jac_str = sim.built_model.generate_julia_diffeq(
  input_parameter_order=input_parameter_order, 
  dae_type=dae_type, 
  get_consistent_ics_solver=pybamm.CasadiSolver(),
  preallocate=preallocate,
  cache_type="standard",
  generate_jacobian=generate_jacobian
);
#=
open("f.jl","w") do io
  write(io,fn_str)
end

open("f_jac.jl","w") do io
  write(io,jac_str)
end

include("../f_jac.jl")
=#
u0 = vec(u0.evaluate())

jac_fn! = runtime_eval(Meta.parse(jac_str));
fn! = runtime_eval(Meta.parse(fn_str))

prob,cbs = get_dae_problem(sim,dae_type="semi-explicit",cache_type="dual")
#sol = solve(prob, Rodas5(autodiff=false), save_everystep=false);

u0 = deepcopy(prob.u0)
du0 = similar(u0)
p = prob.p
t = 0.0
prob.f(du0,u0,p,t)
prob_symbolic,cbs = get_dae_problem(sim,dae_type="semi-explicit",cache_type="symbolic")
jac_sparsity = float(Symbolics.jacobian_sparsity((du,u)->prob_symbolic.f(du,u,p,t),du0,u0))

f = deepcopy(prob.f.f)

func_sparse = ODEFunction{true,true}(f;jac_prototype=jac_sparsity,mass_matrix=sparse(prob.f.mass_matrix))


prob_sparse = ODEProblem(func_sparse,u0,prob.tspan)

function incompletelu(W,du,u,p,t,newW,Plprev,Prprev,solverdata)
    if newW === nothing || newW
      Pl = ilu(convert(AbstractMatrix,W), τ = 5000.0)
    else
      Pl = Plprev
    end
    Pl,nothing
  end


Base.eltype(::IncompleteLU.ILUFactorization{Tv,Ti}) where {Tv,Ti} = Tv


alg = QBDF

sol = solve(prob_sparse,alg(concrete_jac=true))

# Calculate voltage in Julia
V = get_variable(sim, sol, "Terminal voltage [V]")
t = get_variable(sim, sol, "Time [s]")


sol_pybamm = sim.solve(t)
V_pybamm = get(sol_pybamm, "Terminal voltage [V]").data

println("Test result $(all(isapprox.(V_pybamm, V, atol=1e-3)))")

@benchmark solve($prob_sparse, alg(linsolve=KLUFactorization(),concrete_jac=true),save_everystep=false)

#SOLVERS:
# - Trapezoid: 35ms, passes tol test
# - ROS3P: 34ms, passes tol test
# - ROS34PW2: 25ms, doesn't pass tol test, looks close to me
# - ImplicitEuler: 45ms, passes tol test
# - QNDF1: 40ms, passes tol test
# - QBDF: 27ms, passes tol test 
# - QBDF1: 40ms, passes tol test
# - QBDF2: 36ms, passes tol test
# - QNDF: 30ms, doesn't pass tol test, close to me except last point
# - ABDF2: 38ms, passes tol test
# - FBDF: 30ms, passes tol test
# - Rodas4P2: 53ms, passes tol test
# - 
