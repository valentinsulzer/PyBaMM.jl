#
# Functions to extract and evaluate variables from a simulation solution
#

function get_variable(sim, sol::T, var_name::String, inputs) where {T<:DESolution}
    input_parameter_order = isnothing(inputs) ? Array[] : collect(keys(inputs))
    p = isnothing(inputs) ? nothing : collect(values(inputs))

    # Generate the function using PyBaMM
    var_converter = pybamm2julia.JuliaConverter(input_parameter_order=input_parameter_order)
    var_converter.convert_tree_to_intermediate(sim.built_model.variables[var_name])
    var_str = var_converter.build_julia_code(funcname="var_func")
    var_func! = runtime_eval(Meta.parse(pyconvert(String,var_str)))

    # Evaluate and fill in the vector
    # 0D variables only for now
    var = Array{Float64}(undef, length(sol.t))
    out = [0.0]
    for i in 1:length(sol.t)
        # Updating 'out' in-place
        var_func!(out, sol.u[i], p, sol.t[i])
        var[i] = out[1]
    end
    return var
end

get_variable(sim, sol, var_name::String) = get_variable(sim, sol, var_name, nothing)

function get_l2loss_function(sim, var_name, inputs, data)
    input_parameter_order = isnothing(inputs) ? nothing : collect(keys(inputs))
    p = isnothing(inputs) ? nothing : collect(values(inputs))

    # Generate the function using PyBaMM
    var_converter = pybamm2julia.JuliaConverter(input_parameter_order=input_parameter_order)
    var_converter.convert_tree_to_intermediate(sim.built_model.variables[var_name])
    var_str = var_converter.build_julia_code(funcname="var_func")
    var_func! = runtime_eval(Meta.parse(pyconvert(String,var_str)))

    # Evaluate L2 loss
    out = [0.0]

    function loss(sol)
        p = sol.prob.p
        sumsq = 0.0
        for i in 1:length(sol.t)
            # Updating 'out' in-place
            var_func!(out, sol.u[i], p, sol.t[i])
            sumsq += (data[i] - out[1])^2
            # @show i,data[i], out[1]
        end
        sumsq
    end
end


function get_variable(sim, sol::Vector, var_name::String, inputs,t)
    input_parameter_order = isnothing(inputs) ? Array[] : collect(keys(inputs))
    p = isnothing(inputs) ? nothing : collect(values(inputs))

    # Generate the function using PyBaMM
    var_converter = pybamm2julia.JuliaConverter(input_parameter_order=input_parameter_order)
    var_converter.convert_tree_to_intermediate(sim.built_model.variables[var_name])
    var_str = var_converter.build_julia_code(funcname="var_func!")
    var_func! = runtime_eval(Meta.parse(pyconvert(String,var_str)))

    # Evaluate and fill in the vector
    # 0D variables only for now
    out = [0.0]
    var_func!(out,sol,t)
    return out[1]
end

function get_variable_function(sim,var_name;inputs=nothing)
    input_parameter_order = isnothing(inputs) ? Array[] : collect(keys(inputs))
    p = isnothing(inputs) ? nothing : collect(values(inputs))

    # Generate the function using PyBaMM
    var_converter = pybamm2julia.JuliaConverter(input_parameter_order=input_parameter_order)
    var_converter.convert_tree_to_intermediate(sim.built_model.variables[var_name])
    var_str = var_converter.build_julia_code(funcname="var_func!")
    var_func! = runtime_eval(Meta.parse(pyconvert(String,var_str)))
    return var_func!
end
