using PyBaMM

pybamm = PyBaMM.pybamm
pack = PyBaMM.pack
pybamm2julia = PyBaMM.pybamm2julia
setup_circuit = PyBaMM.setup_circuit

Np = 10
Ns = 1
curr = 18.0
p = nothing 
t = 0.0
functional = true
options = Dict("thermal" => "lumped","SEI"=>"reaction limited")
model = pybamm.lithium_ion.DFN(name="DFN", options=options)
netlist = setup_circuit.setup_circuit(Np, Ns, I=curr)
experiment = pybamm.Experiment(repeat(["Discharge at 1.8 A until 10.5 V","Charge at 1.8 A until 12.6 V"],100))   

pybamm_pack = pack.Pack(model, netlist, functional=functional, thermal=true, operating_mode = experiment, top_bc = "symmetry", right_bc = "symmetry")
pybamm_pack.build_pack()

timescale = pyconvert(Float64,pybamm_pack.timescale.evaluate())
cellconverter = pybamm2julia.JuliaConverter(cache_type = "symbolic", inplace=true)
cellconverter.convert_tree_to_intermediate(pybamm_pack.cell_model)
cell_str = cellconverter.build_julia_code()
cell_str = pyconvert(String, cell_str)
cell! = eval(Meta.parse(cell_str))


myconverter = pybamm2julia.JuliaConverter(cache_type = "symbolic")
myconverter.convert_tree_to_intermediate(pybamm_pack.pack)
pack_str = myconverter.build_julia_code()

forcing_function = pybamm_pack.forcing_functions[0]
forcing_converter = pybamm2julia.JuliaConverter(cache_type = "symbolic")
forcing_converter.convert_tree_to_intermediate(forcing_function)
forcing_str = forcing_converter.build_julia_code()
forcing_str = pyconvert(String,forcing_str)

forcing_function = eval(Meta.parse(forcing_str))
open("cvpack.jl","w") do io
    println(io, pack_str)
end


icconverter = pybamm2julia.JuliaConverter(override_psuedo = true)
icconverter.convert_tree_to_intermediate(pybamm_pack.ics)
ic_str = icconverter.build_julia_code()

u0 = eval(Meta.parse(pyconvert(String,ic_str)))
jl_vec = u0()

pack_str = pyconvert(String, pack_str)
jl_func = eval(Meta.parse(pack_str))

dy = similar(jl_vec)

jac_sparsity = float(Symbolics.jacobian_sparsity((du,u)->jl_func(du,u,p,t),dy,jl_vec))

cellconverter = pybamm2julia.JuliaConverter(cache_type = "dual", inplace=true)
cellconverter.convert_tree_to_intermediate(pybamm_pack.cell_model)
cell_str = cellconverter.build_julia_code()
cell_str = pyconvert(String, cell_str)
cell! = eval(Meta.parse(cell_str))

myconverter = pybamm2julia.JuliaConverter(cache_type = "dual")
myconverter.convert_tree_to_intermediate(pybamm_pack.pack)
pack_str = myconverter.build_julia_code()

forcing_function = pybamm_pack.forcing_functions[0]
forcing_converter = pybamm2julia.JuliaConverter(cache_type = "dual")
forcing_converter.convert_tree_to_intermediate(forcing_function)
forcing_str = forcing_converter.build_julia_code()
forcing_str = pyconvert(String,forcing_str)

forcing_function = eval(Meta.parse(forcing_str))

pack_voltage_index = Np + 1
pack_voltage = 3.5.*Ns
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

integrator = 0

func = ODEFunction(jl_func, mass_matrix=mass_matrix, jac_prototype=jac_sparsity)
prob = ODEProblem(func, jl_vec, (0.0, 3600/timescale), nothing)
integrator = init(prob, QNDF(concrete_jac=true))



@showprogress "cycling..." for i in 1:length(experiment.operating_conditions)
    forcing_function = pybamm_pack.forcing_functions[i-1]
    termination_function = pybamm_pack.termination_functions[i-1]

    forcing_converter = pybamm2julia.JuliaConverter(cache_type = "dual")
    forcing_converter.convert_tree_to_intermediate(forcing_function)
    forcing_str = forcing_converter.build_julia_code()
    forcing_str = pyconvert(String,forcing_str)

    forcing_function = eval(Meta.parse(forcing_str))
    
    termination_converter = pybamm2julia.JuliaConverter(cache_type = "dual")
    termination_converter.convert_tree_to_intermediate(termination_function)
    termination_str = termination_converter.build_julia_code()
    termination_str = pyconvert(String, termination_str)

    termination_function = eval(Meta.parse(termination_str))

    done = false
    start_t = integrator.t
    while !done
        step!(integrator)
        done = any((Base.@invokelatest termination_function(integrator.u, (integrator.t - start_t)*pyconvert(Float64, pybamm_pack.built_model.timescale.evaluate()))).<0)
    end
end

saved_vars = get_pack_variables(pybamm_pack, integrator.sol, ["Current [A]","Terminal voltage [V]"])
