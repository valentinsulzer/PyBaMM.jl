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
  "Negative electrode porosity" => Dict("mean"=>0.5,"stddev"=>0.1,"name"=>"neg_por"),
  "Positive electrode porosity" => Dict("mean"=>0.3, "stddev"=>0.1,"name"=>"pos_por")
)

distribution_params = pydict(distribution_params)
    
pybamm_pack = pybamm_pack.Pack(model, netlist, functional=functional, thermal=true)
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
    "Positive electrolyte concentration [mol.m-3]",
    "Separator electrolyte concentration [mol.m-3",
    "Negative electrolyte concentration [mol.m-3]",
    "Current [A]"
]


saved_vars = Dict{String, Any}(var_of_interest=>[] for var_of_interest in vars_of_interest)
for var_of_interest in vars_of_interest
    expr = 0
    if var_of_interest == "Current [A]"
        base_expr = pybamm_pack.batteries[battery]["cell"].children[1].children[0]
        expr = pybamm2julia.PybammJuliaFunction([sv],base_expr,"f",false)
    else
        base_expr = pybamm2julia.PybammJuliaFunction([sv],pybamm_pack.built_model.variables[var_of_interest],"f",false)
        expr = pycopy.deepcopy(base_expr)
        offset = pybamm_pack.batteries[battery]["offset"]
        offsetter = pack.offsetter(pybamm_pack.batteries[battery]["offset"])
        offsetter.add_offset_to_state_vectors(expr.expr)
    end
    tv_converter = pybamm2julia.JuliaConverter()
    tv_converter.convert_tree_to_intermediate(expr)
    tv_str = tv_converter.build_julia_code()

    tv_str = pyconvert(String,tv_str)
    tv = eval(Meta.parse(tv_str))
    test_eval = Base.@invokelatest tv(sol[:,1])
    size_return = size(test_eval)

    this_arr = zeros(Np*Ns,length(sol.t),size_return[1])

    for (i,battery) in enumerate(pybamm_pack.batteries)
        expr = 0
        if var_of_interest == "Current [A]"
            base_expr = pybamm_pack.batteries[battery]["cell"].children[1].children[0]
            expr = pybamm2julia.PybammJuliaFunction([sv],base_expr,"f",false)
        else
            base_expr = pybamm2julia.PybammJuliaFunction([sv],pybamm_pack.built_model.variables[var_of_interest],"f",false)
            expr = pycopy.deepcopy(base_expr)
            offset = pybamm_pack.batteries[battery]["offset"]
            offsetter = pack.offsetter(pybamm_pack.batteries[battery]["offset"])
            offsetter.add_offset_to_state_vectors(expr.expr)
        end

        tv_converter = pybamm2julia.JuliaConverter()
        tv_converter.convert_tree_to_intermediate(expr)
        tv_str = tv_converter.build_julia_code()

        tv_str = pyconvert(String,tv_str)
        tv = eval(Meta.parse(tv_str))
    
        for (j, t) in enumerate(sol.t)
            V = tv(sol[:,j])
            this_arr[i,j,:] .= V
        end
    end
    saved_vars[var_of_interest] = this_arr
end
