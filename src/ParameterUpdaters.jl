

"""
    DirectDescent <: ParameterUpdater
     
"""
struct DirectDescent <: ParameterUpdater
end

init(pu::DirectDescent,pinit) = pinit
update(pu::DirectDescent,p,g::Function) = -g(p)




"""
    AdaGrad <: ParameterUpdater
     
"""
mutable struct AdaGrad <: ParameterUpdater
    G_t::Vector{Float64}
    
    AdaGrad() = new()
end


function init(pu::AdaGrad,pinit) 
    pu.G_t = zeros(eltype(pinit),length(pinit))
    pinit
end
function update(pu::AdaGrad,p,g::Function)
    g_t = g(p)
    pu.G_t += g_t.^2

    -1./ (sqrt.(pu.G_t + 1e-8)) .* g_t  
end