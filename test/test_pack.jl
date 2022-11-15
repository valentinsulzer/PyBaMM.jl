using Test
using PyBaMM



@testset "run pack" begin
  pybamm = PyBaMM.pybamm
  pybamm_pack = PyBaMM.pybamm_pack
  pybamm2julia = PyBaMM.pybamm2julia
  setup_circuit = PyBaMM.setup_circuit

  Np = 3
  Ns = 3
  curr = 1.8
  p = nothing 
  t = 0.0
  functional = true
  options = Dict("thermal" => "lumped")
  model = pybamm.lithium_ion.DFN(name="DFN", options=options)
  netlist = setup_circuit.setup_circuit(Np, Ns, I=curr)   
  pack = pybamm_pack.Pack(model, netlist, functional=functional, thermal=true)
  pack.build_pack()
  timescale = pyconvert(Float64,pack.timescale.evaluate())
  cellconverter = pybamm2julia.JuliaConverter(cache_type = "symbolic", inplace=true)
  cellconverter.convert_tree_to_intermediate(pack.cell_model)
  cell_str = cellconverter.build_julia_code()
  cell_str = pyconvert(String, cell_str)
  cell! = eval(Meta.parse(cell_str))

  myconverter = pybamm2julia.JuliaConverter(cache_type = "symbolic")
  myconverter.convert_tree_to_intermediate(pack.pack)
  pack_str = myconverter.build_julia_code()

  ics_vector = pack.ics
  np_vec = ics_vector.evaluate()
  jl_vec = vec(pyconvert(Array{Float64}, np_vec))

  pack_str = pyconvert(String, pack_str)
  jl_func = eval(Meta.parse(pack_str))

  dy = similar(jl_vec)

  jac_sparsity = float(Symbolics.jacobian_sparsity((du,u)->jl_func(du,u,p,t),dy,jl_vec))

  cellconverter = pybamm2julia.JuliaConverter(cache_type = "dual", inplace=true)
  cellconverter.convert_tree_to_intermediate(pack.cell_model)
  cell_str = cellconverter.build_julia_code()
  cell_str = pyconvert(String, cell_str)
  cell! = eval(Meta.parse(cell_str))

  myconverter = pybamm2julia.JuliaConverter(cache_type = "dual")
  myconverter.convert_tree_to_intermediate(pack.pack)
  pack_str = myconverter.build_julia_code()

  ics_vector = pack.ics
  np_vec = ics_vector.evaluate()
  jl_vec = vec(pyconvert(Array{Float64}, np_vec))

  pack_voltage_index = Np + 1
  pack_voltage = 1.0
  jl_vec[1:Np] .=  curr
  jl_vec[pack_voltage_index] = pack_voltage

  pack_str = pyconvert(String, pack_str)
  jl_func = eval(Meta.parse(pack_str))

  #build mass matrix.
  pack_eqs = falses(pyconvert(Int,pack.len_pack_eqs))

  cell_rhs = trues(pyconvert(Int,pack.len_cell_rhs))
  cell_algebraic = falses(pyconvert(Int,pack.len_cell_algebraic))
  cells = repeat(vcat(cell_rhs,cell_algebraic),pyconvert(Int, pack.num_cells))
  differential_vars = vcat(pack_eqs,cells)
  mass_matrix = sparse(diagm(differential_vars))
  func = ODEFunction(jl_func, mass_matrix=mass_matrix,jac_prototype=jac_sparsity)
  prob = ODEProblem(func, jl_vec, (0.0, 3600/timescale), nothing)

  sol = solve(prob, Trapezoid(linsolve=KLUFactorization(),concrete_jac=true))

  @test size(Array(sol))[1] > 1
  @test size(Array(sol))[2] > 1

end
