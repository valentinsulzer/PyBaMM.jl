#
# Functions to extract and evaluate variables from a simulation solution
#

function get_variable(sim, sol, var_name::String, input_parameter_order, p)
    # Generate the function using PyBaMM
    pybamm = pyimport("pybamm")
    var_str = pybamm.get_julia_function(
        sim.built_model.variables[var_name],
        funcname="var_func",
        input_parameter_order=input_parameter_order
    )
    var_func! = runtime_eval(Meta.parse(var_str))

    # Evaluate and fill in the vector
    # 0D variables only for now
    var = Array{Float64}(undef, length(sol.t))
    out = [0.0]
    for idx in 1:length(sol.t)
        # Updating 'out' in-place
        var_func!(out, sol.u[idx], p, sol.t[idx])
        var[idx] = out[1]
    end
    return var
end

get_variable(sim, sol, var_name) = get_variable(sim, sol, var_name, nothing, nothing)