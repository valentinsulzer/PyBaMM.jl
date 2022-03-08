#
# Functions to extract and evaluate variables from a simulation solution
#
using OrdinaryDiffEq

function get_ode_problem(sim, tend, inputs)
    
    input_parameter_order = isnothing(inputs) ? nothing : collect(keys(inputs))
    p = isnothing(inputs) ? nothing : collect(values(inputs))
    
    sim.build()
    rhs_str, u0_str = sim.built_model.generate_julia_diffeq(input_parameter_order=input_parameter_order)
    
    # PyBaMM-generated functions
    sim_rhs! = runtime_eval(Meta.parse(rhs_str))
    sim_u0! = runtime_eval(Meta.parse(u0_str))

    # Evaluate initial conditions
    u0 = Array{Float64}(undef, sim.built_model.concatenated_rhs.shape[1])
    sim_u0!(u0, p)

    # Scale the time
    tau = sim.built_model.timescale.evaluate()
    tspan = (0, tend/tau)
    
    # Create problem, isinplace is explicitly true as cannot be inferred from
    # runtime_eval function
    prob = ODEProblem{true}(sim_rhs!, u0, tspan, p)
    return prob

end

# Defaults

get_ode_problem(sim, tend::Real) = get_ode_problem(sim, tend, nothing)
get_ode_problem(sim, inputs::AbstractDict) = get_ode_problem(sim, 3600, inputs)
get_ode_problem(sim) = get_ode_problem(sim, 3600, nothing)