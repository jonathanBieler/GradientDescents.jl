"""
    SimpleTracer <: Tracer
    
Only remembers the last value.
"""
mutable struct SimpleTracer <: Tracer
    p
    p_previous
end

function init!(t::SimpleTracer,p,η)
    t.p = p 
    t.p_previous = p      
end
function update!(t::SimpleTracer,p,η)
    t.p_previous = t.p
    t.p = p 
end


"""
    FullTracer <: Tracer
    
Remembers all.    
"""
mutable struct FullTracer <: Tracer
    p::Vector{Float64}
    p_previous::Vector{Float64}
    ps::Vector{Vector{Float64}}
    η::Vector{Float64}
    
    FullTracer() = new()
end

function init!(t::FullTracer,p,η)
    t.p = p 
    t.p_previous = p
    t.ps = Vector{Float64}[]
    t.η = Float64[]
end
function update!(t::FullTracer,p,η)
    t.p_previous = t.p
    t.p = p 
    push!(t.ps,p)
    push!(t.η,η)
end


