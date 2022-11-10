function get_experiment_probs(sim; dae_type = "semi-explicit", initial_soc = nothing)
    #tspan is a lie!
    sim.build_for_experiment(initial_soc = initial_soc)
    p = nothing

    #I'm only going to support time-based stuff for now.
    op_conds = sim.experiment.operating_conditions
    num_steps = length(op_conds)
    probs = []
    for step in 1:num_steps
        op_cond = op_conds[step-1] # pythoncall uses python indexing
        built_model = sim.op_conds_to_built_models[op_cond["string"]]
        real_model = pybamm.numpy_concatenation(
            built_model.concatenated_rhs,
            built_model.concatenated_algebraic
        )
        input_parameter_order = []
        preallocate = true
        cache_type = "standard"

        if dae_type=="implicit"
            fn_str, u0_str = built_model.generate_julia_diffeq(
                input_parameter_order=input_parameter_order, 
                dae_type=dae_type, 
                get_consistent_ics_solver=pybamm.CasadiSolver(),
                preallocate=preallocate,
                cache_type=cache_type
            )
        else
            fn_str, u0_str = built_model.generate_julia_diffeq(
                input_parameter_order=input_parameter_order, 
                dae_type=dae_type, 
                get_consistent_ics_solver=nothing,
                preallocate=preallocate,
                cache_type=cache_type
            )
        end

        fn_str = string(fn_str)
        u0_str = string(u0_str)
    
        # PyBaMM-generated functions
        sim_fn! = runtime_eval(Meta.parse(fn_str))
        u0_fn! = runtime_eval(Meta.parse(u0_str))
        
    
    
        # Evaluate initial conditions
        len_y = convert(Int, pyconvert(Int,built_model.len_rhs_and_alg))
        u0 = zeros(len_y)
        u0_fn!(u0,p)

        len_rhs = convert(Int, pyconvert(Int,built_model.len_rhs))
        len_alg = convert(Int, pyconvert(Int,built_model.len_alg))


        tau = pyconvert(Float64,built_model.timescale.evaluate())
        tspan = (0, 3600 ./tau)

        if len_alg>0
            differential_vars = Bool.(vcat(ones(len_rhs), zeros(len_alg)))
            if dae_type=="implicit"
                prob = DAEProblem{true}(sim_fn!, du0, u0, tspan, p, differential_vars=differential_vars)
            elseif dae_type=="semi-explicit"
                mass_matrix = diagm(differential_vars)
                if cache_type=="gpu"
                    mass_matrix=cu(mass_matrix)
                end
                func! = ODEFunction{true,true}(sim_fn!, mass_matrix=mass_matrix)
                # Create problem, isinplace is explicitly true as cannot be inferred from
                # runtime_eval function
                prob = ODEProblem{true}(func!, u0, tspan, p,initializealg=BrownFullBasicInit())
            else
                println(dae_type)
            end
        else
            prob = ODEProblem{true}(sim_fn!, u0, tspan, p)
        end


        push!(probs, prob)
    end
    return probs
end

function get_termination_condition(sim)
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

function solve_with_experiment(sim; alg=Trapezoid, autodiff=false, vars_to_save = [], save_everystep=true)
    probs = get_experiment_probs(sim, initial_soc = 1.0)
    termination_funcs = get_termination_condition(sim)
    steps = length(sim.experiment.operating_conditions)
    sols = []
    ts = []
    var_funcs = []
    save_vars = Dict(var_to_save => Float64[] for var_to_save in vars_to_save)
    dt = 0.0
    ic = probs[1].u0
    for step in 1:steps
        println("doing step $step")
        prob = probs[step]
        term_cond = termination_funcs[step]
        built_model = sim.op_conds_to_built_models[sim.experiment.operating_conditions[step-1]["string"]]
        
        for (i,var_to_save) in enumerate(vars_to_save)
            var_func = pybamm.PybammJuliaFunction([pybamm.StateVector(pyslice(0,1)), pybamm.Time()], built_model.variables[var_to_save], "jl_var_func", false)
            julia_converter = pybamm.JuliaConverter()
            julia_converter.convert_tree_to_intermediate(var_func)
            jl_str = julia_converter.build_julia_code()
            jl_func = eval(Meta.parse(pyconvert(String,jl_str)))
            push!(var_funcs, jl_func)
        end

        if step!=1
            #set initial condition for the next step.
            dt = ts[end]
            new_model = sim.op_conds_to_built_models[sim.experiment.operating_conditions[step-1]["string"]]
            old_model = sim.op_conds_to_built_models[sim.experiment.operating_conditions[step-2]["string"]]
            new_u0 = deepcopy(prob.u0)
            old_u0 = ic
            set_initial_conditions!(new_u0, old_model, new_model, old_u0)
            prob = remake(prob, u0=new_u0)
        end
        done = false
        integrator = init(prob, alg(autodiff=autodiff), reltol=1e-3, abstol=1e-3, save_everystep=save_everystep)
        while !done
            step!(integrator)
            done = any((Base.@invokelatest term_cond(integrator.u, integrator.t*pyconvert(Float64, built_model.timescale.evaluate()))).<0)
            for (i,var_to_save) in enumerate(vars_to_save)
                push!(save_vars[var_to_save], (Base.@invokelatest var_funcs[i](integrator.u, integrator.t*pyconvert(Float64, built_model.timescale.evaluate())))[1])
            end
            push!(ts, integrator.t*pyconvert(Float64, built_model.timescale.evaluate()).+dt)
        end
        ic = integrator.u
    end
    return sols, save_vars, ts
end



