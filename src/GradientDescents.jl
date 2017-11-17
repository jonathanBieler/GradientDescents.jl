module GradientDescents

    import Base: done, next

    abstract type GradientProvider end
    abstract type ParameterUpdater end
    abstract type LearningRateUpdater end
    abstract type TraceRecorder end
    abstract type Terminator end

    type GradientDescent{T <: GradientProvider, K <: ParameterUpdater, Q <: LearningRateUpdater, P <: Terminator, L <: TraceRecorder}
        gradient::T
        p_updater::K
        η_updater::Q
        terminator::P
        trace::L
        callback::Function
    end

    function optimize(gd::GradientDescent, pinit, opt)

        η = init(gd.η_updater)
        p = init(gd.p_updater,pinit)
        init!(gd.trace,p,η)
        init!(gd.terminator)

        for i=1:opt.niter

            f,g = next(gd.gradient)
            η += update(gd.η_updater,η)
            p += η*update(gd.p_updater,p,g)

            update!(gd.trace,p,η)

            done(gd.terminator,f,gd.trace) && return
            gd.callback(p,f)
        end
        gd.trace
    end

    # sub types

    struct StaticGradient{T<:Function, K <: Function} <: GradientProvider
        f::K
        g::T
    end
    next(g::StaticGradient) = (g.f, g.g)
        
    struct DirectDescent <: ParameterUpdater
    end
    init(sd::DirectDescent,pinit) = pinit
    update(sd::DirectDescent,p,g::Function) = -g(p)

    struct ConstLearningRate <: LearningRateUpdater
    end
    init(lr::ConstLearningRate) = 1e-2
    update(lr::ConstLearningRate,η) = zero(η)

    mutable struct OneTraceRecorder <: TraceRecorder
        p
        p_previous
    end
    function init!(t::OneTraceRecorder,p,η)
        t.p = p 
        t.p_previous = p      
    end
    function update!(t::OneTraceRecorder,p,η)
        t.p_previous = t.p
        t.p = p 
    end

    struct SimpleTerminator <: Terminator
    end
    init!(t::SimpleTerminator) = nothing
    done(t::SimpleTerminator,f,trace) = false

end # module

using BBOBFunctions, ForwardDiff

G = GradientDescents

f = BBOBFunctions.F1.f
g = p -> ForwardDiff.gradient(f,p)

gd = G.GradientDescent(
    G.StaticGradient(f,g),
    G.DirectDescent(),
    G.ConstLearningRate(),
    G.SimpleTerminator(),
    G.OneTraceRecorder(rand(3),rand(3)),
    (p,f) -> println(f(p))
)

type Opt
    niter
end

G.optimize(gd, rand(3), Opt(1000)).p ≈ BBOBFunctions.F1.x_opt[1:3]