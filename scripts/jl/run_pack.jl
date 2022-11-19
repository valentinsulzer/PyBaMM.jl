using PyBaMM
using LinearSolve
using TerminalLoggers
using JLD2
using Plots
plotly()

pybamm = pyimport("pybamm")
pybamm2julia = pyimport("pybamm2julia")
setup_circuit = pyimport("setup_circuit")
pack = pyimport("pack")


Np = 3
Ns = 3

curr = 0.6*Np

p = nothing 
t = 0.0
functional = true

options = Dict("thermal" => "lumped")


model = pybamm.lithium_ion.DFN(name="DFN", options=options)

netlist = setup_circuit.setup_circuit(Np, Ns, I=curr)
    
    pybamm_pack = pack.Pack(model, netlist, functional=functional, thermal=true, left_bc = "symmetry")
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

    ics_vector = pybamm_pack.ics
    np_vec = ics_vector.evaluate()
    jl_vec = vec(pyconvert(Array{Float64}, np_vec))

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
    open("pack20x10.jl","w") do io
	println(io,pack_str)
    end
    open("cell20x10.jl","w") do io
	println(io,cell_str)
    end

    ics_vector = pybamm_pack.ics
    np_vec = ics_vector.evaluate()
    jl_vec = vec(pyconvert(Array{Float64}, np_vec))

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

