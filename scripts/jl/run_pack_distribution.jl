using PyBaMM
using LinearSolve
using TerminalLoggers
using DiffEqOperators
using JLD2


pybamm = pyimport("pybamm")
pybamm2julia = pyimport("pybamm2julia")
setup_circuit = pyimport("setup_circuit")
pybamm_pack = pyimport("pack")

Np = 3
Ns = 3

curr = 1.8

p = nothing 
t = 0.0
functional = true

options = Dict("thermal" => "lumped")


model = pybamm.lithium_ion.DFN(name="DFN", options=options)

netlist = setup_circuit.setup_circuit(Np, Ns, I=curr)

distribution_params = Dict(
  "Negative electrode porosity" => Dict("mean"=>0.5,"stddev"=>0.1,"name"=>"neg_por")
)

distribution_params = pydict(distribution_params)
    
pybamm_pack = pybamm_pack.Pack(model, netlist, functional=functional, thermal=true, distribution_params = distribution_params)
#pybamm_pack = pybamm_pack.Pack(model, netlist, functional=functional, thermal=true)
pybamm_pack.build_pack()


timescale = pyconvert(Float64,pybamm_pack.timescale.evaluate())


if functional
    cellconverter = pybamm2julia.JuliaConverter(cache_type = "symbolic", inplace=true)
    cellconverter.convert_tree_to_intermediate(pybamm_pack.cell_model)
    cell_str = cellconverter.build_julia_code()
    cell_str = pyconvert(String, cell_str)
    cell! = eval(Meta.parse(cell_str))
else
    cell_str = ""
end

myconverter = pybamm2julia.JuliaConverter(cache_type = "symbolic")
myconverter.convert_tree_to_intermediate(pybamm_pack.pack)
pack_str = myconverter.build_julia_code()

icconverter = pybamm2julia.JuliaConverter(override_psuedo = true)
icconverter.convert_tree_to_intermediate(pybamm_pack.ics)
ic_str = icconverter.build_julia_code()

u0 = eval(Meta.parse(pyconvert(String,ic_str)))
jl_vec = u0()

pack_str = pyconvert(String, pack_str)
jl_func = eval(Meta.parse(pack_str))

dy = similar(jl_vec)


println("Calculating Jacobian Sparsity")
jac_sparsity = float(Symbolics.jacobian_sparsity((du,u)->jl_func(du,u,p,t),dy,jl_vec))


if functional
    cellconverter = pybamm2julia.JuliaConverter(cache_type = "dual", inplace=true)
    cellconverter.convert_tree_to_intermediate(pybamm_pack.cell_model)
    cell_str = cellconverter.build_julia_code()
    cell_str = pyconvert(String, cell_str)
    cell! = eval(Meta.parse(cell_str))
else
    cell_str = ""
end

myconverter = pybamm2julia.JuliaConverter(cache_type = "dual")
myconverter.convert_tree_to_intermediate(pybamm_pack.pack)
pack_str = myconverter.build_julia_code()

pack_voltage_index = Np + 1
pack_voltage = 1.0
jl_vec[1:Np] .=  curr
jl_vec[pack_voltage_index] = pack_voltage

pack_str = pyconvert(String, pack_str)
jl_func = eval(Meta.parse(pack_str))

#build mass matrix.
pack_eqs = falses(pyconvert(Int,pybamm_pack.len_pack_eqs))

cell_rhs = trues(pyconvert(Int,pybamm_pack.len_cell_rhs))
cell_algebraic = falses(pyconvert(Int,pybamm_pack.len_cell_algebraic))
cells = repeat(vcat(cell_rhs,cell_algebraic),pyconvert(Int, pybamm_pack.num_cells))
differential_vars = vcat(pack_eqs,cells)
mass_matrix = sparse(diagm(differential_vars))
func = ODEFunction(jl_func, mass_matrix=mass_matrix,jac_prototype=jac_sparsity)
prob = ODEProblem(func, jl_vec, (0.0, 3600/timescale), nothing)

using IncompleteLU
function incompletelu(W,du,u,p,t,newW,Plprev,Prprev,solverdata)
if newW === nothing || newW
Pl = ilu(convert(AbstractMatrix,W), τ = 50.0)
else
Pl = Plprev
end
Pl,nothing
end


sol = solve(prob, Trapezoid(linsolve=KLUFactorization(),concrete_jac=true))