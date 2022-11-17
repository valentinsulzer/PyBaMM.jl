function get_pack_variables(pybamm_pack, sol, vars_of_interest)
    sv = pybamm.StateVector(pyslice(0,1))
    saved_vars = Dict{String, Any}(var_of_interest=>[] for var_of_interest in vars_of_interest)
    for var_of_interest in vars_of_interest
        set_distribution_params = pylist(pybamm_pack.batteries["V0"]["distribution parameters"])
        expr = 0
        if var_of_interest == "Current [A]"
            base_expr = pybamm_pack.batteries["V0"]["cell"].children[1].children[0]
            expr = pybamm2julia.PybammJuliaFunction([sv],base_expr,"f",false)
        else
            base_expr = pybamm2julia.PybammJuliaFunction([sv],pybamm_pack.built_model.variables[var_of_interest],"f",false)
            expr = pycopy.deepcopy(base_expr)
            offset = pybamm_pack.batteries["V0"]["offset"]
            offsetter = pack.offsetter(pybamm_pack.batteries["V0"]["offset"])
            offsetter.add_offset_to_state_vectors(expr.expr)
        end
        for param in set_distribution_params
            param.set_psuedo(expr.expr, pybamm_pack.batteries["V0"]["distribution parameters"][param])
        end

        tv_converter = pybamm2julia.JuliaConverter(override_psuedo = true)
        tv_converter.convert_tree_to_intermediate(expr)
        tv_str = tv_converter.build_julia_code()

        tv_str = pyconvert(String,tv_str)
        tv = eval(Meta.parse(tv_str))
        test_eval = Base.@invokelatest tv(sol[:,1])
        size_return = size(test_eval)

        this_arr = zeros(length(pybamm_pack.batteries),length(sol.t),size_return[1])

        for (i,battery) in enumerate(pybamm_pack.batteries)
            set_distribution_params = pybamm_pack.batteries[battery]["distribution parameters"]

            expr = 0
            if var_of_interest == "Current [A]"
                base_expr = pybamm_pack.batteries[battery]["cell"].children[1].children[0]
                expr = pybamm2julia.PybammJuliaFunction([sv],base_expr,"f",false)
            else
                base_expr = pybamm2julia.PybammJuliaFunction([sv],pybamm_pack.built_model.variables[var_of_interest],"f",false)
                expr = pycopy.deepcopy(base_expr)
                offset = pybamm_pack.batteries[battery]["offset"]
                offsetter = pack.offsetter(pybamm_pack.batteries[battery]["offset"])
                offsetter.add_offset_to_state_vectors(expr.expr)
            end

            for param in set_distribution_params
                param.set_psuedo(expr.expr, pybamm_pack.batteries[battery]["distribution parameters"][param])
            end

            tv_converter = pybamm2julia.JuliaConverter(override_psuedo = true)
            tv_converter.convert_tree_to_intermediate(expr)
            tv_str = tv_converter.build_julia_code()

            tv_str = pyconvert(String,tv_str)
            tv = eval(Meta.parse(tv_str))
        
            for (j, t) in enumerate(sol.t)
                V = Base.@invokelatest tv(sol[:,j])
                this_arr[i,j,:] .= V
            end
        end
        saved_vars[var_of_interest] = this_arr
    end
    return saved_vars
end
