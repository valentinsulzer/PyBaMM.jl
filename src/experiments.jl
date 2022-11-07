function get_experiment_probs(sim, inputs, tend)
    #tspan is a lie!
    sim.build_for_experiment()

    #I'm only going to support time-based stuff for now.
    op_conds = sim.experiment.operating_conditions
    num_steps = length(op_conds)
    funcs = []
    for step in 1:num_steps
        op_cond = op_conds[step-1] # pythoncall uses python indexing
        built_model = sim.op_conds_to_built_models[op_cond["electric"]]
        real_model = pybamm.numpy_concatenation(
            built_model.concatenated_rhs,
            built_model.concatenated_algebraic
        )
        jl_converter = pybamm.JuliaConverter()
        jl_converter.convert_tree_to_intermediate(real_model)
        jl_str = pyconvert(String,jl_converter.build_julia_code())
        jl_func = eval(Meta.parse(jl_str))
        push!(funcs, jl_func)
    end
    return funcs
end

function get_termination_condition(sim, inputs, tend)
    #tspan is a lie!
    sim.build_for_experiment()

    #I'm just going to do this myself.
    op_conds = sim.experiment.operating_conditions
    num_steps = length(op_conds)
    exprs = []
    event_inputs = []

    funcs = []
    for step in 1:num_steps
        exprs = []
        sv = pybamm.StateVector(pyslice(0,1))
        op_cond = op_conds[step-1]
        
        #Check for Time First.
        max_t = pybamm.Scalar(op_cond["time"])
        t = pybamm.Time()
        pybamm_time_expr = max_t - t

        push!(exprs, pybamm_time_expr)
        push!(event_inputs, t)

        #Check for Current Second.
        if "Current cut-off [A]" in op_cond
            current = pybamm.AbsoluteValue(sim.op_conds_to_built_models[op_cond["string"]].variables["Current [A]"])
            max_current = pybamm.Scalar(op_cond["Current cut-off [A]"])
            pybamm_current_expr = current - max_current
            push!(exprs, pybamm_current_expr)
        end

        #Check for Voltage Third.
        if "Voltage cut-off [V]" in op_cond
            voltage = sim.op_conds_to_built_models[op_cond["string"]].variables["Terminal voltage [V]"]
            if pyconvert(String, op_cond["type"]) == "power"
                inp = pyconvert(Float64, op_cond["Power input [W]"])
            elseif pyconvert(String, op_cond["type"]) == "current"
                inp = pyconvert(Float64, op_cond["Current input [A]"])
            else
                print(pyconvert(String, op_cond["type"]))
                error("Unknown type")
            end

            sign_jl = sign(inp)
            sign_py = pybamm.Scalar(sign_jl)

            voltage_expr = sign_py * (voltage - pybamm.Scalar(op_cond["Voltage cut-off [V]"]))
            push!(exprs, voltage_expr)
        end
        out_expr = pybamm.numpy_concatenation(exprs...)
        out_expr = pybamm.PybammJuliaFunction([sv, t], out_expr, "check_for_termination", false)
        
        func_converter = pybamm.JuliaConverter()
        func_converter.convert_tree_to_intermediate(out_expr)
        jl_str = pyconvert(String, func_converter.build_julia_code())
        jl_func = eval(Meta.parse(jl_str))
        push!(funcs, jl_func)
    end
    return funcs
end

function set_initial_conditions!(new_u, old_model, new_model, last_u)
    fp = pybamm.PsuedoInputParameter("fp")
    a = pybamm.StateVector(pyslice(0,1))
    for ic in new_model.initial_conditions
        
        pybamm_expr = pybamm.PybammJuliaFunction([a, fp],old_model.variables[ic.name] + fp,"jl_func", false)
        
        julia_converter = pybamm.JuliaConverter()
        julia_converter.convert_tree_to_intermediate(pybamm_expr)
        jl_str = julia_converter.build_julia_code()
        
        jl_func = eval(Meta.parse(pyconvert(String,jl_str)))

        if length(new_model.variables[ic.name].y_slices) > 1
            @error "variables can only hold 1 y slice."
        end
        slice = new_model.variables[ic.name].y_slices[0]
        start_index = pyconvert(Int, slice.start) + 1
        stop_index = pyconvert(Int, slice.stop)
        sl_step = pyconvert(Any, slice.step)
        if sl_step === nothing
            new_u[start_index:stop_index] .= Base.@invokelatest jl_func(last_u, 0)
        else
            step = pyconvert(Int, slice.step)
            new_u[start_index:step:stop_index] .= Base.@invokelatest jl_func(last_u, 0)
        end
    end    
end


