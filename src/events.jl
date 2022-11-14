#
# Functions to handle pybamm event handling in julia
#
using OrdinaryDiffEq

function build_callback(event,size)
    type = pyconvert(Int,event.event_type.value)
    scalar_check = pyconvert(String,event.expression.__class__.__name__)
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
    myconverter = pybamm2julia.JuliaConverter()
    myconverter.convert_tree_to_intermediate(event.expression)
    jl_str = pyconvert(String,myconverter.build_julia_code())
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