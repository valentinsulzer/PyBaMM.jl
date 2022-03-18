#
# Functions to handle pybamm event handling in julia
#
using DiffEqBase

function build_callback(event,size)
    type = event.event_type.value
    scalar_check = event.expression.__class__.__name__
    if scalar_check=="Scalar"
        return nothing
    end
    if type==0
        jl_event = build_terminating_callback(event,size)
    elseif type==3
        jl_event = build_terminating_callback(event,size)
    else
        error("unrecognized callback")
    end
    return jl_event
end

function build_terminating_callback(event,size)
    #Generate Julia Function with the Event
    pybamm = pyimport("pybamm")
    jl_str = pybamm.get_julia_function(event.expression)
    jl_func! = runtime_eval(Meta.parse(jl_str))
    #Generate Condition for Callback
    f = begin
    f2 = let cs = (
        cache_du = zeros(size),
    )
    function f3(u,p,t)
    jl_func!(cs.cache_du,u,p,t)
    return cs.cache_du[1]
    end
    end
    end
    callback = ContinuousCallback(f,terminate!)
    return callback
end