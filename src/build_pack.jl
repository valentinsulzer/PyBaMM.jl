
function build_pack(Np, Ns, model, tend=3600.0)
    
    netlist = lp.setup_circuit(Np, Ns)
    
    pybamm_pack = pybamm.Pack(model, netlist)
    pybamm_pack.build_pack()

    timescale = pyconvert(Float64,pybamm_pack.timescale.evaluate())
    
    myconverter = pybamm.JuliaConverter()
    myconverter.convert_to_intermediate(pybamm_pack.pack)
    jl_str = myconverter.build_julia_code()

    ics_vector = pybamm_pack.ics
    np_vec = ics_vector.evaluate()
    jl_vec = pyconvert(Vector{Float64}, np_vec)

    jl_func = eval(Meta.parse(jl_str))

    #build mass matrix.
    pack_eqs = falses(pyconvert(Int,pybamm_pack.len_pack_eqs))

    cell_rhs = trues(pyconvert(Int,pybamm_pack.len_cell_rhs))
    cell_algebraic = falses(trues(pyconvert(Int,pybamm_pack.len_cell_algebraic)))
    cells = repeat(vcat(cell_rhs,cell_algebraic),pyconvert(Int, pybamm_pack.num_cells))
    differential_vars = vcat(pack_eqs,cells)
    mass_matrix = diagm(differential_vars)
    func = ODEFunction(jl_func, mass_matrix=mass_matrix)
    prob = ODEProblem(func, jl_vec, (0.0, 3600/timescale), nothing)
    return prob
end
