module GradientDescents

    using ForwardDiff

    import Base: done, next

    abstract type GradientProvider end
    abstract type ParameterUpdater end
    abstract type LearningRateUpdater end
    abstract type Tracer end
    abstract type Terminator end

    type GradientDescent{T <: GradientProvider, K <: ParameterUpdater, Q <: LearningRateUpdater, P <: Terminator, L <: Tracer}
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
    include("GradientProviders.jl")
    include("ParameterUpdaters.jl")
    include("LearningRateUpdater.jl")
    include("Tracers.jl")
    include("Terminators.jl")
       
end # module




