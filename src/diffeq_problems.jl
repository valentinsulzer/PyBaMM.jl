#
# Functions to create DiffEq problems
#
using OrdinaryDiffEq

function _problem_setup(sim, tend, inputs;dae_type="implicit",preallocate=true,cache_type="standard",)
    pybamm = pyimport("pybamm")
    
    input_parameter_order = isnothing(inputs) ? Array[] : collect(keys(inputs))
    p = isnothing(inputs) ? nothing : collect(values(inputs))

    sim.build()
    if dae_type=="implicit"
        fn_str, u0 = sim.built_model.generate_julia_diffeq(
            input_parameter_order=input_parameter_order, 
            dae_type=dae_type, 
            get_consistent_ics_solver=pybamm.CasadiSolver(),
            preallocate=preallocate,
            cache_type=cache_type
        )
    else
        fn_str, u0 = sim.built_model.generate_julia_diffeq(
            input_parameter_order=input_parameter_order, 
            dae_type=dae_type, 
            get_consistent_ics_solver=nothing,
            preallocate=preallocate,
            cache_type=cache_type
        )
    end

    fn_str = string(fn_str)

    # PyBaMM-generated functions
    sim_fn! = runtime_eval(Meta.parse(fn_str))

    # Evaluate initial conditions
    len_y = convert(Int, pyconvert(Int,sim.built_model.len_rhs_and_alg))
    u0 = vec(pyconvert(Array{Float64,2},u0.evaluate()))
    if cache_type=="gpu"
        u0 = cu(u0)
    end

    # Scale the time
    tau = pyconvert(Float64,sim.built_model.timescale.evaluate())
    tspan = (0, tend/tau)
    

    # Create callbacks
    ncb = length(u0)
    callbacks = [build_callback(event,ncb) for event in sim.built_model.events]
    callbacks = callbacks[findall(.!isnothing.(callbacks))]
    callbackSet = CallbackSet(callbacks...)

    sim_fn!, u0, tspan, p, callbackSet
end

function get_ode_problem(sim, tend, inputs;preallocate=true,cache_type="standard")
    sim_fn!, u0, tspan, p, callbackSet = _problem_setup(sim, tend, inputs,preallocate=preallocate,cache_type=cache_type)
    
    # Create problem, isinplace is explicitly true as cannot be inferred from
    # runtime_eval function
    ODEProblem{true}(sim_fn!, u0, tspan, p),callbackSet
end

# Defaults
get_ode_problem(sim, tend::Real;preallocate=true,cache_type="standard") = get_ode_problem(sim, tend, nothing,preallocate=preallocate,cache_type=cache_type)
get_ode_problem(sim, inputs::AbstractDict;preallocate=true,cache_type="standard") = get_ode_problem(sim, 3600, inputs,preallocate=preallocate,cache_type=cache_type)
get_ode_problem(sim;preallocate=true,cache_type="standard") = get_ode_problem(sim, 3600, nothing,preallocate=preallocate,cache_type=cache_type)

function get_dae_problem(sim, tend, inputs;dae_type="implicit",preallocate=true,cache_type="standard")
    sim_fn!, u0, tspan, p, callbackSet = _problem_setup(sim, tend, inputs,dae_type=dae_type,preallocate=preallocate,cache_type=cache_type)
    
    # Create vector of 1s and 0s to indicate differential and algebraic variables
    len_rhs = convert(Int, pyconvert(Int,sim.built_model.len_rhs))
    len_alg = convert(Int, pyconvert(Int,sim.built_model.len_alg))
    differential_vars = Bool.(vcat(ones(len_rhs), zeros(len_alg)))
    
    
    # Create problem, isinplace is explicitly true as cannot be inferred from
    # runtime_eval function
    if dae_type == "implicit"
        du0 = zeros(size(u0))
        return DAEProblem{true}(sim_fn!, du0, u0, tspan, p, differential_vars=differential_vars),callbackSet
    else
        mass_matrix = diagm(differential_vars)
        if cache_type=="gpu"
            mass_matrix=cu(mass_matrix)
        end
        func! = ODEFunction{true,true}(sim_fn!, mass_matrix=mass_matrix)
        # Create problem, isinplace is explicitly true as cannot be inferred from
        # runtime_eval function
        return ODEProblem{true}(func!, u0, tspan, p,initializealg=BrownFullBasicInit()),callbackSet
    end

end

# Defaults
get_dae_problem(sim, tend::Real;dae_type="implicit",preallocate=true,cache_type="standard") = get_dae_problem(sim, tend, nothing,dae_type=dae_type,preallocate=preallocate,cache_type=cache_type)
get_dae_problem(sim, inputs::AbstractDict;dae_type="implicit",preallocate=true,cache_type="standard") = get_dae_problem(sim, 3600, inputs,dae_type=dae_type,preallocate=preallocate,cache_type=cache_type)
get_dae_problem(sim;dae_type="implicit",preallocate=true,cache_type="standard") = get_dae_problem(sim, 3600, nothing,dae_type=dae_type,preallocate=preallocate,cache_type=cache_type)


function get_optimized_problem(sim;tend=3600.0,inputs=nothing)
    #Generate the dual problem to be used
    prob,cbs = get_dae_problem(sim,tend,inputs,dae_type="semi-explicit",cache_type="dual")
    
    # Generate a Jacobian Prototype
    prob_symbolic,cbs = get_dae_problem(sim,tend,inputs,dae_type="semi-explicit",cache_type="symbolic")
    u0 = deepcopy(prob_symbolic.u0)
    du0 = zeros(length(u0))
    p = deepcopy(prob_symbolic.p)
    t = 0.0
    jac_sparsity = float(Symbolics.jacobian_sparsity((du,u)->prob_symbolic.f(du,u,p,t),du0,u0))

    #Now tie it together
    f = deepcopy(prob.f.f)
    func_sparse = ODEFunction{true,true}(f;jac_prototype=jac_sparsity,mass_matrix=sparse(prob.f.mass_matrix))
    prob_sparse = ODEProblem(func_sparse,u0,prob.tspan,p)

    #And return an integrator that the user can use!
    return prob_sparse
end
