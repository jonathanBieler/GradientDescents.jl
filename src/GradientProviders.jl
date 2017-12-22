###################
## StaticGradient

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


###################
## MiniBatch

    mutable struct MiniBatch{T<:Function, K <: Function,P} <: GradientProvider
            f::K
            g::T
            idx::Vector{Vector{Int}}
            data::P
            batch_size::Int
    end
    
    function mini_batch_idx(data, batch_size)
        p = randperm(length(data))
        idx = Vector{Int}[]
        while length(p) >= batch_size
            push!(idx, splice!(p,1:batch_size))
        end
        !isempty(p) && push!(idx, p)
        
        idx
    end

    function MiniBatch(f::Function,data,batch_size)
        
        idx = mini_batch_idx(data, batch_size)
        g = p -> p#place holder
        
        MiniBatch(f,g,idx,data,batch_size)
    end

    function next(m::MiniBatch) 
        
        if isempty(m.idx)
            m.idx = mini_batch_idx(m.data, m.batch_size)
        end 
        
        idx = splice!(m.idx,1)
        
        f = p -> m.f(p,m.data[idx])
        g = p -> ForwardDiff.gradient(f,p)
        
        (f, g)
    end