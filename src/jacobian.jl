#
# Functions to build an analytical jacobian for pybamm
#
function generate_jacobian(sim; input_parameter_order = [], cache_type="standard",preallocate=true)
    sim.build()
    built_model = sim.built_model
    name = pyconvert(String,built_model.name.replace(" ", "_"))
    size_state = built_model.concatenated_initial_conditions.size
    state_vector = pybamm.StateVector(pyslice(0,size_state))
    expr = pybamm.numpy_concatenation(
        built_model.concatenated_rhs,
        built_model.concatenated_algebraic
    ).jac(state_vector)
    converter = pybamm2julia.JuliaConverter(
        input_parameter_order = input_parameter_order,
        cache_type = cache_type,
        inline=true,
        preallocate=preallocate,
    )
    converter.convert_tree_to_intermediate(expr)
    jac_str = converter.build_julia_code(funcname="jac_"*name)
    jac_fn! = runtime_eval(Meta.parse(pyconvert(String,jac_str)))
    return jac_fn!
end