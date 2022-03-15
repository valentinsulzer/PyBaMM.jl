#
# Functions to handle pybamm event handling in julia
#
using DiffEqBase

function build_callback(event)
    type = event.event_type.value
    scalar_check = event.expression.__class__.__name__
    if scalar_check=="Scalar"
        return nothing
    end
    if type==0
        jl_event = build_terminating_callback(event)
    elseif type==1
        jl_event = build_discontinuity_callback(event)
    elseif type==2
        jl_event = build_interpolant_extrapolation_callback(event)
    elseif type==3
        jl_event = build_switch_callback(event)
    else
        error("unrecognized callback")
    end
    return jl_event
end

function build_terminating_callback(event)
    jl_func = runtime_eval(Meta.parse(pybamm.get_julia_function(event.expression)))
end