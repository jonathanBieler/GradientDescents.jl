
struct StaticGradient{T<:Function, K <: Function} <: GradientProvider
        f::K
        g::T
end

function StaticGradient(f::Function)
    g = p -> ForwardDiff.gradient(f,p)
    StaticGradient(f,g)
end

next(g::StaticGradient) = (g.f, g.g)

#TODO use ForwardDiff to compute both