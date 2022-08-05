struct DiffCache{T <: AbstractArray, S <: AbstractArray}
    du::T
    sym_du::S
end
get_tmp(dc::DiffCache, u::Number) = dc.du
get_tmp(dc::DiffCache, u::AbstractArray) = dc.du
get_tmp(dc::DiffCache, u::AbstractArray{Num}) = dc.sym_du
symcache(du::AbstractArray, sym_du::AbstractArray) = DiffCache(du, sym_du)
