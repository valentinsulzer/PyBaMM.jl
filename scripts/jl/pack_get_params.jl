using PyBaMM
using LinearSolve
using TerminalLoggers
using DiffEqOperators
using JLD2


pybamm = pyimport("pybamm")
pybamm2julia = pyimport("pybamm2julia")
setup_circuit = pyimport("setup_circuit")
pack = pyimport("pack")

Np = 2
Ns = 2

curr = 1.2

p = nothing 
t = 0.0
functional = true

options = Dict("thermal" => "lumped")


model = pybamm.lithium_ion.DFN(name="DFN", options=options)

netlist = setup_circuit.setup_circuit(Np, Ns, I=curr)

distribution_params = Dict(
  "Negative electrode porosity" => Dict("mean"=>0.5,"stddev"=>0.1,"name"=>"neg_por"),
  "Positive electrode porosity" => Dict("mean"=>0.3, "stddev"=>0.1,"name"=>"pos_por")
)

distribution_params = pydict(distribution_params)
    
pybamm_pack = pack.Pack(model, netlist, functional=functional, thermal=true, distribution_params = distribution_params)
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


sol = solve(prob, Trapezoid(linsolve=KLUFactorization(),concrete_jac=true))

pycopy = pyimport("copy")

sv = pybamm.StateVector(pyslice(0,1))

vars_of_interest = [
    "Terminal voltage [V]",
    "Cell temperature [K]", 
    "Current [A]",
    "Positive electrolyte concentration [mol.m-3]",
    "Separator electrolyte concentration [mol.m-3]",
    "Negative electrolyte concentration [mol.m-3]",
    "Positive particle concentration [mol.m-3]",
    "Negative particle concentration [mol.m-3]",
    "Positive electrolyte potential [V]",
    "Separator electrolyte potential [V]",
    "Negative electrolyte potential [V]",
    "Positive electrode potential [V]",
    "Negative electrode potential [V]",
    "x [m]",
    "x_p [m]",
    "x_s [m]",
    "x_n [m]",
    "r_p [m]",
    "r_n [m]"
]

saved_vars = get_pack_variables(pybamm_pack, sol, vars_of_interest)
