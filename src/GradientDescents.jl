module GradientDescents

    using ForwardDiff, Optim

    import Optim: Optimizer
    import Base: done, next

    abstract type GradientProvider end
    abstract type ParameterUpdater end
    abstract type LearningRateUpdater end
    abstract type Tracer end
    abstract type Terminator end

    type GradientDescent{T <: GradientProvider, K <: ParameterUpdater, Q <: LearningRateUpdater, P <: Terminator, L <: Tracer} <: Optimizer
        gradient::T
        p_updater::K
        η_updater::Q
        terminator::P
        tracer::L
        callback::Function
    end

    function optimize(gd::GradientDescent, pinit, opt)

        #get gradient first for initialization
        f,g = next(gd.gradient)
        g_t = g(pinit)
        
        p, η = init(gd.p_updater,pinit), init(gd.η_updater,g_t,pinit)
        
        init!(gd.p_updater.momentum,p)
        init!(gd.tracer,p,η)
        init!(gd.terminator)

        for i=1:opt.niter

            η += update(gd.η_updater,η,g_t)
            p += update(gd.p_updater,η,p,g_t)

            update!(gd.p_updater.momentum,g_t)
            update!(gd.tracer,p,η)

            done(gd.terminator,f,g_t,gd.tracer) && return gd.tracer
            gd.callback(p,f)
            
            #get next gradient
            f,g = next(gd.gradient)
            g_t = g(p)
        end
        gd.tracer
    end

    # sub types
    include("GradientProviders.jl")
    include("ParameterUpdaters.jl")
    include("LearningRateUpdater.jl")
    include("Tracers.jl")
    include("Terminators.jl")
       
end # module




