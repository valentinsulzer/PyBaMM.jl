using PyCall,PyBaMM,BenchmarkTools,DifferentialEquations,SparseArrays,LinearSolve,Symbolics,IncompleteLU,Plots



pybamm = pyimport("pybamm")
model = pybamm.lithium_ion.DFN(name="DFN")
sim = pybamm.Simulation(model)

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

sol = solve(prob_sparse,alg(linsolve=KLUFactorization(),concrete_jac=true))

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
