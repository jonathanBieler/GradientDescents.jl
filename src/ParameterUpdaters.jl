abstract type Momentum end


function init!(m::Momentum,pinit) 
    m.previous_gradient = zeros(size(pinit)) 
end
function update!(m::Momentum,g_t) 
    m.previous_gradient = g_t
end

#############
## NoMomentum

    mutable struct NoMomentum <: Momentum
        previous_gradient::Vector{Float64}
    end 
    
    NoMomentum() = NoMomentum(Float64[])
    
    update!(m::NoMomentum,g_t)  = nothing 
    momentum(m::NoMomentum,g_t) = m.previous_gradient #previous_gradient is always zero

################
## SimpleMomentum

    mutable struct SimpleMomentum <: Momentum
        γ::Float64
        previous_gradient::Vector{Float64}
    end 

    SimpleMomentum() = SimpleMomentum(1e-6,Float64[])
    SimpleMomentum(γ::Float64) = SimpleMomentum(γ,Float64[])

    momentum(m::SimpleMomentum,g_t) = -m.γ * m.previous_gradient


#################
## DirectDescent


"""
    DirectDescent <: ParameterUpdater
     
"""
struct DirectDescent{T <: Momentum} <: ParameterUpdater
    momentum::T
end

DirectDescent() = DirectDescent(NoMomentum())

init(u::DirectDescent,pinit) = pinit
update(u::DirectDescent,η,p,g_t) = -η*g_t + momentum(u.momentum,g_t)

#############
## AdaGrad

"""
    AdaGrad <: ParameterUpdater
     
"""
mutable struct AdaGrad{T <: Momentum} <: ParameterUpdater
    momentum::T
    G_t::Vector{Float64}
end
AdaGrad() = AdaGrad(NoMomentum(),Vector{Float64}())
AdaGrad(m::Momentum) = AdaGrad(m,Vector{Float64}())

function init(u::AdaGrad,pinit) 
    u.G_t = zeros(eltype(pinit),length(pinit))
    pinit
end
function update(u::AdaGrad,η,p,g_t)
    
    u.G_t += g_t.^2

    -η*1./sqrt.(u.G_t + 1e-8) .* g_t  + momentum(u.momentum,g_t)
end

