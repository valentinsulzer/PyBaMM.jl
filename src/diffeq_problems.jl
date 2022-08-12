#
# Functions to create DiffEq problems
#
using OrdinaryDiffEq

function _problem_setup(sim, tend, inputs;dae_type="implicit",preallocate=true,cache_type="standard")
    pybamm = pyimport("pybamm")
    
    input_parameter_order = isnothing(inputs) ? nothing : collect(keys(inputs))
    p = isnothing(inputs) ? nothing : collect(values(inputs))

    sim.build()
    if dae_type=="implicit"
        fn_str, u0_str = sim.built_model.generate_julia_diffeq(
            input_parameter_order=input_parameter_order, 
            dae_type=dae_type, 
            get_consistent_ics_solver=pybamm.CasadiSolver(),
            preallocate=preallocate,
            cache_type=cache_type
        )
    else
        fn_str, u0_str = sim.built_model.generate_julia_diffeq(
            input_parameter_order=input_parameter_order, 
            dae_type=dae_type, 
            get_consistent_ics_solver=nothing,
            preallocate=preallocate,
            cache_type=cache_type
        )
    end

    # PyBaMM-generated functions
    sim_fn! = runtime_eval(Meta.parse(fn_str))
    sim_u0! = runtime_eval(Meta.parse(u0_str))

    # Evaluate initial conditions
    len_y = convert(Int, sim.built_model.len_rhs_and_alg)
    u0 = Array{Float64}(undef, len_y)
    if cache_type=="gpu"
        u0 = cu(u0)
    end

    sim_u0!(u0, p)

    # Scale the time
    tau = sim.built_model.timescale.evaluate()
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
    len_rhs = convert(Int, sim.built_model.len_rhs)
    len_alg = convert(Int, sim.built_model.len_alg)
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
