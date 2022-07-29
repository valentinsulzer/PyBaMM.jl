#
# Functions to create DiffEq problems
#
using OrdinaryDiffEq

function _problem_setup(sim, tend, inputs)
    pybamm = pyimport("pybamm")
    
    input_parameter_order = isnothing(inputs) ? nothing : collect(keys(inputs))
    p = isnothing(inputs) ? nothing : collect(values(inputs))
    
    sim.build()
    fn_str, u0_str = sim.built_model.generate_julia_diffeq(
        input_parameter_order=input_parameter_order, 
        dae_type="implicit", 
        get_consistent_ics_solver=pybamm.CasadiSolver()
    )
    
    # PyBaMM-generated functions
    sim_fn! = runtime_eval(Meta.parse(fn_str))
    sim_u0! = runtime_eval(Meta.parse(u0_str))

    # Evaluate initial conditions
    len_y = convert(Int, sim.built_model.len_rhs_and_alg)
    u0 = Array{Float64}(undef, len_y)
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

function get_ode_problem(sim, tend, inputs)
    sim_fn!, u0, tspan, p, callbackSet = _problem_setup(sim, tend, inputs)
    
    # Create problem, isinplace is explicitly true as cannot be inferred from
    # runtime_eval function
    ODEProblem{true}(sim_fn!, u0, tspan, p),callbackSet
end

# Defaults
get_ode_problem(sim, tend::Real) = get_ode_problem(sim, tend, nothing)
get_ode_problem(sim, inputs::AbstractDict) = get_ode_problem(sim, 3600, inputs)
get_ode_problem(sim) = get_ode_problem(sim, 3600, nothing)

function get_dae_problem(sim, tend, inputs)
    sim_fn!, u0, tspan, p, callbackSet = _problem_setup(sim, tend, inputs)
    
    # Create vector of 1s and 0s to indicate differential and algebraic variables
    len_rhs = convert(Int, sim.built_model.len_rhs)
    len_alg = convert(Int, sim.built_model.len_alg)
    differential_vars = vcat(ones(len_rhs), zeros(len_alg))
    
    
    # Create problem, isinplace is explicitly true as cannot be inferred from
    # runtime_eval function
    du0 = zeros(size(u0))
    DAEProblem{true}(sim_fn!, du0, u0, tspan, p, differential_vars=differential_vars),callbackSet
end

# Defaults
get_dae_problem(sim, tend::Real) = get_dae_problem(sim, tend, nothing)
get_dae_problem(sim, inputs::AbstractDict) = get_dae_problem(sim, 3600, inputs)
get_dae_problem(sim) = get_dae_problem(sim, 3600, nothing)