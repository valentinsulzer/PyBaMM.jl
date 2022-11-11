using PyBaMM
using LinearSolve
using TerminalLoggers
using DiffEqOperators
using JLD2

pybamm = pyimport("pybamm")
lp = pyimport("liionpack")


Np = 10
Ns = 10

curr = 6.5

p = nothing 
t = 0.0
functional = true

options = Dict("thermal" => "lumped")


model = pybamm.lithium_ion.DFN(name="DFN", options=options)

netlist = lp.setup_circuit(Np, Ns, I=curr)
    
pybamm_pack = pybamm.Pack(model, netlist, functional=functional, thermal=true)
pybamm_pack.build_pack()

timescale = pyconvert(Float64,pybamm_pack.timescale.evaluate())


if functional
    cellconverter = pybamm.JuliaConverter(inplace=true, cache_type="dual")
    cellconverter.convert_tree_to_intermediate(pybamm_pack.cell_model)
    cell_str = cellconverter.build_julia_code()
    cell_str = pyconvert(String, cell_str)
    cell! = eval(Meta.parse(cell_str))
else
    cell_str = ""
end
    

myconverter = pybamm.JuliaConverter()
myconverter.convert_tree_to_intermediate(pybamm_pack.pack)
pack_str = myconverter.build_julia_code()

open("bigpack.jl","w") do io
    println(io,pack_str)
end

ics_vector = pybamm_pack.ics
np_vec = ics_vector.evaluate()
jl_vec = vec(pyconvert(Array{Float64}, np_vec))

pack_str = pyconvert(String, pack_str)
jl_func = eval(Meta.parse(pack_str))

dy = similar(jl_vec)

pack_voltage_index = Np + 1
pack_voltage = 1.0
jl_vec[1:Np] .=  curr
jl_vec[pack_voltage_index] = pack_voltage

pack_eqs = falses(pyconvert(Int,pybamm_pack.len_pack_eqs))

cell_rhs = trues(pyconvert(Int,pybamm_pack.len_cell_rhs))
cell_algebraic = falses(pyconvert(Int,pybamm_pack.len_cell_algebraic))
cells = repeat(vcat(cell_rhs,cell_algebraic),pyconvert(Int, pybamm_pack.num_cells))
differential_vars = vcat(pack_eqs,cells)
mass_matrix = sparse(diagm(differential_vars))

using IncompleteLU
Base.eltype(::IncompleteLU.ILUFactorization{Tv,Ti}) where {Tv,Ti} = Tv
function incompletelu(W,du,u,p,t,newW,Plprev,Prprev,solverdata)
    if newW === nothing || newW
      Pl = ilu(convert(AbstractMatrix,W), τ = 50.0)
    else
      Pl = Plprev
    end
    Pl,nothing
  end
  

func = ODEFunction(jl_func, mass_matrix=mass_matrix)
prob = ODEProblem(func, jl_vec, (0.0, 3600/timescale), nothing)

sol = solve(prob,Trapezoid(linsolve=KrylovJL_GMRES(),precs=incompletelu))
@time solve(prob,Trapezoid(linsolve=KrylovJL_GMRES(),precs=incompletelu))

sol_arr = Array(sol)
sol_t = Array(sol.t)

@save "10x10pack_krylov.jld2" sol_arr sol_t



