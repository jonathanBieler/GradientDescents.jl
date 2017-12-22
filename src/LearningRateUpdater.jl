
guess_η(g_t,p,α) = α/vecnorm(g_t,1)

######################
## ConstLearningRate

    mutable struct ConstLearningRate <: LearningRateUpdater
        η::Float64
    end
    ConstLearningRate() = ConstLearningRate(0.0)

    function init(lr::ConstLearningRate,g_t,p)
        lr.η = lr.η == 0.0 ? guess_η(g_t,p,1e-1) : lr.η
        lr.η
    end
    update(lr::ConstLearningRate,η,g_t) = zero(η)

#########################
## HypergradientDescent

    #cf. Online Learning Rate Adaptation with Hypergradient Descent
    mutable struct HypergradientDescent <: LearningRateUpdater
        η::Float64
        β::Float64
        previous_grad::Vector{Float64}
        
        HypergradientDescent(η,β) = new(η,β)
    end
    HypergradientDescent() = HypergradientDescent(0.0,0.0)

    function init(lr::HypergradientDescent,g_t,p)
        lr.previous_grad = zeros(eltype(p),length(p))
        lr.η = lr.η == 0.0 ? guess_η(g_t,p,1e-5) : lr.η
        println("guessed η = $(lr.η)")
        lr.β = lr.η/1e4
        lr.η
    end
    function update(lr::HypergradientDescent,η,g_t)  
    
        delta = lr.β * sum(g_t .* lr.previous_grad)
        lr.previous_grad = copy(g_t)
        delta
    end
