pybamm = pyimport("pybamm")

function get_variable(sim, sol, var_name::String, input_parameter_order, p)
    var_str = pybamm.get_julia_function(
        sim.built_model.variables[var_name],
        funcname="var_func",
        input_parameter_order=input_parameter_order
    )
    eval(Meta.parse(var_str))

    var = Array{Float64}(undef, length(sol.t))
    out = [0.0]
    for idx in 1:length(sol.t)
        var_func!(out, sol.u[idx], p, sol.t[idx])
        var[idx] = out[1]
    end
    return var
end

get_variable(sim, sol, var_name) = get_variable(sim, sol, var_name, nothing, nothing)