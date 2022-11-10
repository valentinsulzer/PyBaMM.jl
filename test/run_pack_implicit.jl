using PyBaMM
using LinearSolve
using TerminalLoggers

pybamm = pyimport("pybamm")
lp = pyimport("liionpack")


Np = 10
Ns = 10

curr = 1.2

p = nothing 
t = 0.0
functional = true

options = Dict("thermal" => "lumped")


model = pybamm.lithium_ion.SPM(name="DFN", options=options)

netlist = lp.setup_circuit(Np, Ns, I=curr)
    
pybamm_pack = pybamm.Pack(model, netlist, functional=functional, thermal=true, implicit=true)
pybamm_pack.build_pack()

timescale = pyconvert(Float64,pybamm_pack.timescale.evaluate())

if functional
    cellconverter = pybamm.JuliaConverter(inplace=true, cache_type = "dual")
    cellconverter.convert_tree_to_intermediate(pybamm_pack.cell_model)
    cell_str = cellconverter.build_julia_code()
    cell_str = pyconvert(String, cell_str)
    cell! = eval(Meta.parse(cell_str))
else
    cell_str = ""
end

packconverter = pybamm.JuliaConverter(override_psuedo=true, cache_type = "dual")
packconverter.convert_tree_to_intermediate(pybamm_pack.pack)
pack_str = packconverter.build_julia_code()

ics_vector = pybamm_pack.ics
np_vec = ics_vector.evaluate()
u0 = vec(pyconvert(Array{Float64}, np_vec))
du0 = deepcopy(u0)

pack_voltage_index = Np + 1
pack_voltage = 1.0
u0[1:Np] .=  curr
u0[pack_voltage_index] = pack_voltage

pack_str = pyconvert(String, pack_str)
jl_func = eval(Meta.parse(pack_str))

#build mass matrix.
pack_eqs = falses(pyconvert(Int,pybamm_pack.len_pack_eqs))

cell_rhs = trues(pyconvert(Int,pybamm_pack.len_cell_rhs))
cell_algebraic = falses(pyconvert(Int,pybamm_pack.len_cell_algebraic))
cells = repeat(vcat(cell_rhs,cell_algebraic),pyconvert(Int, pybamm_pack.num_cells))
differential_vars = vcat(pack_eqs,cells)

prob = DAEProblem(jl_func, du0, u0, (0.0, 600/timescale), nothing, differential_vars=differential_vars)


#Haven't gotten GMRES to work
@benchmark solve(prob, IDA())