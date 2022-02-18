#
# Create and discretize the SPMe in pybamm, convert it to a format Julia DiffEq likes, and solve
# A lot of this could eventually be converted into functions in PyBaMM.jl
#

using PyCall

pybamm = pyimport("pybamm")
np = pyimport("numpy")

# load model
model = pybamm.lithium_ion.DFN(name="DFN")

sim = pybamm.Simulation(model)
sim.build()

rhs_str, u0_str = sim.built_model.generate_julia_diffeq()

eval(Meta.parse(rhs_str))
eval(Meta.parse(u0_str))

len_rhs = convert(Int, sim.built_model.len_rhs)
len_alg = convert(Int, sim.built_model.len_alg)
differential_vars = vcat(ones(len_rhs), zeros(len_alg))

u0 = Array{Float64}(undef, len_rhs + len_alg)
DFN_u0!(u0, [])
du0 = zeros(size(u0))
out = Array{Float64}(undef, size(u0))

using BenchmarkTools
# Check that function is not allocating
@btime DFN!(out, du0, u0, 0, 0)


# Solve in Julia
using OrdinaryDiffEq

tend = 3600/sim.built_model.timescale.evaluate()
tspan = (0.0, tend)


prob = DAEProblem(DFN!,du0,u0,tspan,differential_vars=differential_vars)

using Sundials
sol = solve(prob, IDA());#, reltol=1e-6, abstol=1e-6, saveat=tend / 100);
sol.u

# Benchmarks
@btime solve(prob, IDA());

# Calculate voltage in Julia
V_str = pybamm.get_julia_function(sim.built_model.variables["Terminal voltage [V]"], funcname="V")#, input_parameter_order=input_parameter_order)
eval(Meta.parse(V_str))

V = Array{Float64}(undef, length(sol.t))
out = [0.0]
for idx in 1:length(sol.t)
    V!(out, sol.u[idx], [], sol.t[idx])
    V[idx] = out[1]
end

# Solve in python
sol_pybamm = sim.solve(np.linspace(0,3600,100))
V_pybamm = get(sol_pybamm, "Terminal voltage [V]").data

# Plots
using Plots

plot(sol.t, V)
scatter!(sol.t, V_pybamm)